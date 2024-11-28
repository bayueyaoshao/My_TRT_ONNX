/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-18 14:23:42
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 14:07:48
 */

#pragma once
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <filesystem>
#include <io.h>
#include <fstream>
#include <numeric>
//#include "cuda_utils.h"
#include "logging.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "tensor_interface_type.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)




namespace tenser_openex
{
	const static int kGpuId = 0;
	enum class ResizeMethod
	{
		Padding = 0,
		Fill = 1,
		None = 2,

	};
	struct input_params_base
	{
		int input_width = 224;
		int input_height = 224;
		char* input_blob_name = (char*)"input";
		char* output_blob_name = (char* )"output";
		int batch_size = 2;
		int class_nums = 1;
		ResizeMethod resizemethod = ResizeMethod::Fill;
	};

	struct infer_results_base
	{

	};

	template<tensor_interface_type TensorType>
	struct input_params : input_params_base {};

	template<tensor_interface_type TensorType>
	struct infer_results : infer_results_base {};

	template<tensor_interface_type TensorType> 
	class tensor_base
	{
		public:
		 	constexpr static tensor_interface_type Type = TensorType;

			using Ptr = std::shared_ptr<tensor_base<Type>>;

			tensor_base(){};
			tensor_base(std::string engine_file)
			{
				if(load_model(engine_file))
					model_load_success = true;
			}

			~ tensor_base()
			{
				if (this->input_data_host_!=nullptr)
					delete this->input_data_host_;
				if (this->output_data_host_ != nullptr)
					delete this->output_data_host_;
			};
		public:
			//加载模型
			virtual bool load_model(std::string engine_file) 
			{
				if (cudaSetDevice(kGpuId) != cudaSuccess) 
					return false;

				size_t size{0};
				// read files
				std::ifstream file(engine_file, std::ios::binary);
				if (!file.good())
				{
					std::cerr << "read " << engine_file << " error!" << std::endl;
					throw(std::exception("model name error"));
					return false;
				}

				std::vector<char> data;
				try
				{
					file.seekg(0, file.end);
					const auto size = file.tellg();
					file.seekg(0, file.beg);
					data.resize(size);
					file.read(data.data(), size);
				}
				catch (const std::exception& e)
				{
					file.close();
					return false;
				}
				file.close();
				runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
				//initLibNvInferPlugins(&logger_, "");
				if (!runtime_) {
					throw(std::exception("create infer runtime fail"));
					return false;
				}
				engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(data.data(), data.size()));
				if (!engine_) {
					throw(std::exception("deserialize engine fail"));
					return false;
				}
				context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
				if (!context_)
				{
					throw(std::exception("create Execution Context fail"));
					return false;
				}

				auto out_dims = this->engine_->getBindingDimensions(1);
				for (int j = 0; j < out_dims.nbDims; j++) {
					if (out_dims.d[j] > 0)
						this->output_size_ *= out_dims.d[j];
				}
				
				return true;		
			};
			//
			virtual bool detect_image(std::vector<cv::Mat> img_vec) 
			{
				try
				{		
					if (img_vec.size()!=params_.batch_size)
					{
						//throw std::exception("batch size not match image nums");
						return false;
					}
					this->src_img_vec_ = img_vec;
					//for (auto& img : this->src_img_vec_)
					//{
					//	if (img.channels() == 1)
					//	cv::cvtColor(img, img, cv::COLOR_GRAY2BGR); //1111
					//}
					preprocess_image();
					this->input_data_host_ = trans_image();
					this->output_data_host_ = infer();
					results_.clear();
					decode_output();

					return true;
				}
				catch(const std::exception& e)
				{
					return false;
				}
				catch(const cv::Exception& e)
				{
					return false;
				}
			};
			//设置参数
			inline void set_params(input_params<Type> params)
			{
				params_ = params;
			}
			//获取结果
			inline void get_results(std::vector<infer_results<Type>>& results)
			{ 
				results =  results_;
			}
			bool model_load_success = false;
		protected:
			//预处理, 缩放等
			virtual void preprocess_image() 
			{
					cv::Mat re,out;
					float scale_min = 1;
					switch (params_.resizemethod)
					{
					case ResizeMethod::Padding:
						//std::cout << "img_h:" << this->src_img_vec_[0].rows<<","<<this->src_img_vec_[0].cols << std::endl;
						for (auto& img : this->src_img_vec_)
						{
							scale_min = params_.input_width / (img.cols*1.0) < params_.input_height / (img.rows*1.0) ? params_.input_width / (img.cols*1.0) : params_.input_height / (img.rows*1.0);
							scale_ = scale_min;	
							re = cv::Mat(scale_min * img.rows, scale_min * img.cols, CV_8UC3);
							cv::resize(img, re, re.size());
							out = cv::Mat(params_.input_width, params_.input_height, CV_8UC3, cv::Scalar(114, 114, 114));
							re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
							img = out;
						}
						//std::cout << "img_h:" << this->src_img_vec_[0].rows<<","<<this->src_img_vec_[0].cols << std::endl;
						break;
					case ResizeMethod::Fill:
						for (auto& img : this->src_img_vec_)
						{
							cv::resize(img, img, cv::Size(params_.input_width,params_.input_height));
						}
						break;
					case ResizeMethod::None:
						break;		
					default:
						break;
					}
			}
			//图像展开,并归一化,获取其在cpu中的内存
			virtual float* trans_image() 
			{
					int channels = 1;
					this->input_data_host_ = new float[this->src_img_vec_[0].total()* channels *params_.batch_size];
					
					bool is32FC3 = true;
					for (size_t batch_idx = 0; batch_idx < params_.batch_size; batch_idx++)
					{
						int img_h = this->src_img_vec_[batch_idx].rows;
						int img_w = this->src_img_vec_[batch_idx].cols;

						//cv::cvtColor(this->src_img_vec_[batch_idx], this->src_img_vec_[batch_idx], cv::COLOR_BGR2RGB);

						cv::Mat input = src_img_vec_[batch_idx];
						
						for (size_t c = 0; c < channels; c++) 
						{
							for (size_t  h = 0; h < img_h; h++) 
							{
								for (size_t w = 0; w < img_w; w++) 
								{ 
									if (is32FC3)
									{
										float value = (((float)this->src_img_vec_[batch_idx].at<float>(h, w)));
										int pos = (batch_idx * channels * img_w * img_h) + c * img_w * img_h + h * img_w + w;
										this->input_data_host_[pos] = value;
									}
									else
									{
										float value = (((float)this->src_img_vec_[batch_idx].at<cv::Vec3b>(h, w)[c]) / 255.0f);
										int pos = (batch_idx * 3 * img_w * img_h) + c * img_w * img_h + h * img_w + w;
										this->input_data_host_[pos] = value;
									}
								}
							}
						}	
					}
					return this->input_data_host_;
			};
			//执行推理，获取检测结果在cpu中的内存
			virtual float* infer() 
			{
				int channel = 1;
					this->output_data_host_ = new float[this->output_size_ *params_.batch_size];		

					const nvinfer1::ICudaEngine& engine = this->context_->getEngine();
					assert(engine.getNbBindings() == 2);
					void* device_data_buffers[2];

					const int inputIndex = engine.getBindingIndex(params_.input_blob_name);
					assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
					const int outputIndex = engine.getBindingIndex(params_.output_blob_name);
					assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

						// Create GPU buffers on device
					CHECK(cudaMalloc(&device_data_buffers[inputIndex], params_.batch_size * channel * params_.input_height * params_.input_width * sizeof(float)));
					CHECK(cudaMalloc(&device_data_buffers[outputIndex],params_.batch_size * this->output_size_*sizeof(float)));

					// Create stream
					cudaStream_t stream;
					CHECK(cudaStreamCreate(&stream));

					CHECK(cudaMemcpyAsync(device_data_buffers[inputIndex], this->input_data_host_, params_.batch_size * channel * this->params_.input_height * params_.input_width * sizeof(float), cudaMemcpyHostToDevice, stream));
					this->context_->setBindingDimensions(0,nvinfer1::Dims4(params_.batch_size , channel,params_.input_height ,params_.input_width));
					this->context_->enqueueV2(device_data_buffers, stream, nullptr);
					CHECK(cudaMemcpyAsync(this->output_data_host_, device_data_buffers[outputIndex], params_.batch_size * this->output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
					cudaStreamSynchronize(stream);

					cudaStreamDestroy(stream);
					CHECK(cudaFree(device_data_buffers[inputIndex]));
					CHECK(cudaFree(device_data_buffers[outputIndex]));

					return this->output_data_host_;
			};
			//解析结果，检测结果内存中解析模型的推理结果
			virtual void decode_output() = 0;
		protected:
			typename input_params<Type> params_; //输入参数
            typename std::vector<infer_results<Type>> results_; //输出结果
			std::shared_ptr<nvinfer1::IRuntime> runtime_;
			std::shared_ptr<nvinfer1::ICudaEngine> engine_;
			std::shared_ptr<nvinfer1::IExecutionContext> context_;
			int output_size_ = 1; // 初始分配的float数量，会在模型加载是读取更新
			std::vector<cv::Mat> src_img_vec_; //输入图像 
			float* input_data_host_; //cpu中输入图像指针
			float* output_data_host_; //cpu中的推理结果指针
			float scale_ = 1;
			Logger logger_;
	};
}