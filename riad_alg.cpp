#include "riad_alg.h"
#include <QElapsedTimer>
#include<QDebug>
#include"tensorrt_api/tensor_RIAD.h"
cv::dnn::Net m_net_model;
tenser_openex::tensor_RIAD<tenser_openex::tensor_interface_type::tensor_RIAD>::Ptr mytensorclass;
void print_log(std::string log)
{
    std::cout << log.c_str();
}


void loadEngineModel(std::string engine_path, int image_size)
{
    mytensorclass = std::make_shared<tenser_openex::tensor_RIAD<tenser_openex::tensor_interface_type::tensor_RIAD>>(engine_path);
    if (!mytensorclass->model_load_success)
    {
        // std::cout <"模型加载失败"> std::ednl;
        return;
    }
    ////配置模型的输入参数
    tenser_openex::input_params<tenser_openex::tensor_interface_type::tensor_RIAD> classifier_params;
    classifier_params.batch_size = 1;
    classifier_params.class_nums = 1;
    classifier_params.input_height = image_size;
    classifier_params.input_width = image_size;
    classifier_params.input_blob_name = (char*)("input");  // 注意名字和模型中一致
    classifier_params.output_blob_name = (char*)("output");
    ////Fill方法直接对图像resize,Padding方法表示保持原图像比例进行resize
    classifier_params.resizemethod = tenser_openex::ResizeMethod::None;
    ////传入参数并执行, detect_image和get_results函数默认是多batch_size的，在模型输入参数batch_size为1的情况下，也需要将输入图片和输出结果放进vector中
    mytensorclass->set_params(classifier_params);

}

//加载模型
void load_model(std::string onnx_path)
{
    m_net_model = cv::dnn::readNet(onnx_path);
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        print_log("defect-DETECTOR: DNN_TARGET_CUDA");
        m_net_model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        m_net_model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        print_log("defect-DETECTOR: DNN_TARGET_CPU");
        m_net_model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net_model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}


// 输入[0,255]输出[0,1]
cv::Mat normalizeImage(const cv::Mat& inputImage, const cv::Scalar& mean, const cv::Scalar& stdDev)
{
    cv::Mat normalizedImage;
    // 将图像像素值范围从 [0, 255] 缩放到 [0, 1]
    cv::Mat floatImage;
    inputImage.convertTo(floatImage, CV_32FC1, 1.0/255);
    //// 计算归一化的均值和标准差
    //cv::Scalar imageMean = cv::mean(floatImage);
    //cv::Scalar imageStdDev;
    //cv::meanStdDev(floatImage, imageMean, imageStdDev);
    //// 归一化操作：减去指定的均值，然后除以指定的标准差
    //cv::subtract(floatImage, mean, normalizedImage);
    //cv::divide(normalizedImage, stdDev, normalizedImage);
    return floatImage;
}


cv::Mat datastream_f2matc3_f(const cv::Mat& datastream, int image_size)
{
    int img_size = image_size * image_size * 1 * sizeof(float);
    cv::Mat blob_R(image_size, image_size, CV_32F);

    // std::memcpy(mat.data, output_img[0].data, m_model_input_size * m_model_input_size * 3 * sizeof(uchar));
    std::memcpy(blob_R.data, datastream.data, img_size);

    // cv::imshow("blob_", blob_);
    return blob_R;
}

// 输入[0,1]输出[0,255]
cv::Mat denormalization(const cv::Mat& normalizedImage) {
    // 定义均值和标准差
    cv::Scalar mean(0.5, 0.5, 0.5);
    cv::Scalar std(0.5, 0.5, 0.5);
    double scale = 255.0;
    // cv::imshow("aa1", normalizedImage);
    // 深拷贝输入的归一化图像
    cv::Mat denormalizedImage = normalizedImage.clone();

    // 还原归一化的图像数据
    denormalizedImage.convertTo(denormalizedImage, CV_32F); // 将图像转换为浮点类型
    // cv::imshow("aa1", denormalizedImage);
    denormalizedImage = (denormalizedImage.mul(std) + mean); // 还原数据
    // cv::imshow("aa2", denormalizedImage);
    denormalizedImage = denormalizedImage * scale;
    denormalizedImage.convertTo(denormalizedImage, CV_8U); // 转换为 8-bit 无符号整数类型

    return denormalizedImage; //结果[0，255]
}

cv::Mat create_cv32fc3_by_path()
{
    std::string file_name = "D:/bottle_contamination_07.png";
    cv::Mat image_ori = cv::imread(file_name);
    int model_image_size = 320;
    cv::Mat image;
    cv::resize(image_ori, image, cv::Size(model_image_size, model_image_size), 0, 0, cv::INTER_LINEAR);
    /*cv::Mat output_C3;
    image.convertTo(output_C3, CV_32FC3, 1.0 / 255.0);*/
    return image;
}

cv::Mat one_image_inference(cv::Mat normalizedImage, int model_image_size, bool isEngine)
{
    cv::Mat blob_;
    QElapsedTimer timer; // 创建定时器
    timer.start(); // 开始计时
    if (isEngine)
    {
        std::vector<cv::Mat> img_vec;
        img_vec.push_back(normalizedImage);
        std::vector<tenser_openex::infer_results<tenser_openex::tensor_interface_type::tensor_RIAD>> results;
        if (mytensorclass->detect_image(img_vec))
        {
            mytensorclass->get_results(results);
            for (size_t i = 0; i < results.size(); i++)
            {
                blob_ = results[i].reconstruct_image;
            }
        }
    }
    else
    {
        cv::Mat blob;
        std::vector<cv::Mat> output_img;  //存放结果特征
        cv::dnn::blobFromImage(normalizedImage, blob, 1.0, cv::Size(model_image_size, model_image_size));
        //cv::Mat blob_in = datastream_f2matc3_f(blob, model_image_size);
        // show_flolt_mat(blob_in, "blob_in");
        m_net_model.setInput(blob);
       
        m_net_model.forward(output_img, m_net_model.getUnconnectedOutLayersNames());
        
        blob_ = datastream_f2matc3_f(output_img[0], model_image_size);// [0-1]
    }
    qint64 elapsedTime = timer.nsecsElapsed(); // 获取纳秒为单位的持续时间
    qDebug() << "-----Execution time: " << elapsedTime / 1e6 << " ms."; // 转换算成毫秒并输出
    

    cv::Mat denormalizedImage = denormalization(blob_);
    // cv::imshow("denormalizedImage", denormalizedImage);
    // cv::imwrite("D:/C_Recon.bmp", denormalizedImage);
    // cv::waitKey();
    return denormalizedImage;
}

cv::Mat model_inference(std::string image_path, int model_image_size, cv::Mat & recMat, bool isEngine)
{
    cv::Mat scores(model_image_size, model_image_size, CV_32FC1, cv::Scalar(0));
    // image_path = "D:/C_Mask.bmp";
    cv::Mat image_ori = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    cv::Mat image;
    cv::resize(image_ori, image, cv::Size(model_image_size, model_image_size), 0, 0, cv::INTER_LINEAR);
    // // 定义归一化所需的均值和标准差
    cv::Scalar meanValue(0.709, 0.381, 0.224);  // 均值
    cv::Scalar stdDevValue(0.127, 0.079, 0.043); // 标准差
    // // 对调整大小后的图像进行归一化
    cv::Mat normalizedImage = normalizeImage(image, meanValue, stdDevValue);
    cv::Mat ret = one_image_inference(normalizedImage, model_image_size, isEngine);
    //cv::imshow("ret", ret);
    return ret;
}