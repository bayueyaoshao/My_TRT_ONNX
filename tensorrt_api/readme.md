<!--
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-25 09:03:15
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 16:06:39
-->

# tensorrt_api introduction

&ensp;tensorrt api 是深度学习模型基于tensorrt的C++推理接口与部分实现。其中基类tensor_base依据tensoerrt的推理流程定义了相应的函数接口，以及输入参数与输出结果的Base结构体，同时对部分通用接口进行了实现(如模型加载)；此外基于tensor_base具体实现了两个常见网络模型alex和yolo5。

&ensp;该api的设计借鉴了部分ram_alg算法平台中检测项的设计思路，以及PCL中对智能指针的使用亮点。

## 一、engine模型的导出
engine模型导出采用tensorRT自带的trtexec工具进行,将onnx模型转换为engine模型
### 1. 模型的导出
#### onnx模型为静态batch_size
`.\trtexec --onnx=E:\OPENEX\ch26t7sp1c1_class4_v13.onnx --saveEngine=E:\OPENEX\ch26t7sp1c1_class4_v13.engine `
#### onnx模型为动态batch_size
需设置模型最小、最大、以及最佳情况下的输出大小
`.\trtexec --onnx=E:\OPENEX\ch26t7sp1c1_class4_v13.onnx --saveEngine=E:\OPENEX\ch26t7sp1c1_class4_v13.engine --minShapes=inputx:1x3x224x224 --optShapes=inputx:16x3x224x224 --maxShapes=inputx:32x3x224x224` 
#### 查看模型的参数信息
`.\trtexec --loadEngine=ch26t7sp2v64_25200_24_640.engine --verbose`


## 二、调用Demo

```
////读取图像，设置engine模型的路径
std::string file = "E:\\OPENEX\\test1.png";
cv::Mat img1 = cv::imread(file1);
std::string file2 = "E:\\OPENEX\\test2.png";
cv::Mat img2 = cv::imread(file2);
std::string engine_model_file = "E:\\OPENEX\\test.engine";
////创建并读取模型
tenser_openex::tensor_classifier<tenser_openex::tensor_interface_type::tensor_classifier>::Ptr mytensorclass = std::make_shared<tenser_openex::tensor_classifier<tenser_openex::tensor_interface_type::tensor_classifier>>(engine_model_file);
if (!mytensorclass->model_load_success)
{
    std::cout <"模型加载失败"> std::ednl;
    return 0;
}
////配置模型的输入参数
tenser_openex::input_params<tenser_openex::tensor_interface_type::tensor_classifier> classifier_params;
classifier_params.batch_size = 2;
classifier_params.class_nums = 4;
classifier_params.input_height = 224;
classifier_params.input_width = 224;
classifier_params.input_blob_name = "input";
classifier_params.output_blob_name = "output";
////Fill方法直接对图像resize,Padding方法表示保持原图像比例进行resize
classifier_params.resizemethod = tenser_openex::ResizeMethod::Fill; 
////传入参数并执行, detect_image和get_results函数默认是多batch_size的，在模型输入参数batch_size为1的情况下，也需要将输入图片和输出结果放进vector中 
mytensorclass->set_params(classifier_params);
std::vector<cv::Mat> img_vec;
img_vec.push_back(img1);
img_vec.push_back(img2);
std::vector<tenser_openex::classifier_results> results;
if(mytensorclass->detect_image(img_vec))
{
    mytensorclass->get_results(results);
    for (size_t i = 0; i < results.size(); i++)
    {
        std::cout << "class_id:" << results[0].class_id << std::endl;
    }
}
```

## 三、扩展实现自己的模型推理接口
可在tensor_base基础上，实现更多深度学习模型基于tensorrt的推理接口
### 1.tensorrt_api相关内容介绍
#### 枚举类tensor_interface_type
该类是模板基类tensor_base的输入类型参数，作为模板的输入参数
![avatar](./0.png)
tensor_base是tensortrt 推理接口的模板基类。其中包括了输入参数与输出结果基结构体，及其派生模板结构体;模型加载，图像预处理，归一化，推理，解码等相应接口，其中decode_output为抽象接口，其余接口给出了一些通用的实现。
#### 接口基类tensor_base
![avatar](./1.png)
### 2.关键点
具体的实现可参考tensor_calssfier和tensor_yolo5两个例子，有以下关键点
#### 1 添加模板参数:
在枚举类tensor_interface_type中添加自己拟实现的模型推理接口名称，作为模板类和模板结构体的模板参数
#### 2 定义自己模型输入参数与推理结果的结构体
infer_results和input_params分别继承其基结构体。其中infer_results务必依据自己模型的输出定义相应的数据格(若一张图对应输出结果，则需定义在容器中)。input_params则是若input_params_base不存在自己想要传入的参数时，定义此参数。
![avatar](./2.png)
![avatar](./3.png)

#### 3 重写相应的函数接口
大部分函数接口的实现已在基类中完成，通常情况下不同的模型只有解析结果存在差异，因此只有decode_output是纯虚函数。具体是需要从ouput_data_host_解析相应的结果，并将其push到result_中。如果在前期处理阶段存在不同，则自行重写相关接口。
![avatar](./4.png)

