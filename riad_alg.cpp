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
        // std::cout <"ģ�ͼ���ʧ��"> std::ednl;
        return;
    }
    ////����ģ�͵��������
    tenser_openex::input_params<tenser_openex::tensor_interface_type::tensor_RIAD> classifier_params;
    classifier_params.batch_size = 1;
    classifier_params.class_nums = 1;
    classifier_params.input_height = image_size;
    classifier_params.input_width = image_size;
    classifier_params.input_blob_name = (char*)("input");  // ע�����ֺ�ģ����һ��
    classifier_params.output_blob_name = (char*)("output");
    ////Fill����ֱ�Ӷ�ͼ��resize,Padding������ʾ����ԭͼ���������resize
    classifier_params.resizemethod = tenser_openex::ResizeMethod::None;
    ////���������ִ��, detect_image��get_results����Ĭ���Ƕ�batch_size�ģ���ģ���������batch_sizeΪ1������£�Ҳ��Ҫ������ͼƬ���������Ž�vector��
    mytensorclass->set_params(classifier_params);

}

//����ģ��
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


// ����[0,255]���[0,1]
cv::Mat normalizeImage(const cv::Mat& inputImage, const cv::Scalar& mean, const cv::Scalar& stdDev)
{
    cv::Mat normalizedImage;
    // ��ͼ������ֵ��Χ�� [0, 255] ���ŵ� [0, 1]
    cv::Mat floatImage;
    inputImage.convertTo(floatImage, CV_32FC1, 1.0/255);
    //// �����һ���ľ�ֵ�ͱ�׼��
    //cv::Scalar imageMean = cv::mean(floatImage);
    //cv::Scalar imageStdDev;
    //cv::meanStdDev(floatImage, imageMean, imageStdDev);
    //// ��һ����������ȥָ���ľ�ֵ��Ȼ�����ָ���ı�׼��
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

// ����[0,1]���[0,255]
cv::Mat denormalization(const cv::Mat& normalizedImage) {
    // �����ֵ�ͱ�׼��
    cv::Scalar mean(0.5, 0.5, 0.5);
    cv::Scalar std(0.5, 0.5, 0.5);
    double scale = 255.0;
    // cv::imshow("aa1", normalizedImage);
    // �������Ĺ�һ��ͼ��
    cv::Mat denormalizedImage = normalizedImage.clone();

    // ��ԭ��һ����ͼ������
    denormalizedImage.convertTo(denormalizedImage, CV_32F); // ��ͼ��ת��Ϊ��������
    // cv::imshow("aa1", denormalizedImage);
    denormalizedImage = (denormalizedImage.mul(std) + mean); // ��ԭ����
    // cv::imshow("aa2", denormalizedImage);
    denormalizedImage = denormalizedImage * scale;
    denormalizedImage.convertTo(denormalizedImage, CV_8U); // ת��Ϊ 8-bit �޷�����������

    return denormalizedImage; //���[0��255]
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
    QElapsedTimer timer; // ������ʱ��
    timer.start(); // ��ʼ��ʱ
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
        std::vector<cv::Mat> output_img;  //��Ž������
        cv::dnn::blobFromImage(normalizedImage, blob, 1.0, cv::Size(model_image_size, model_image_size));
        //cv::Mat blob_in = datastream_f2matc3_f(blob, model_image_size);
        // show_flolt_mat(blob_in, "blob_in");
        m_net_model.setInput(blob);
       
        m_net_model.forward(output_img, m_net_model.getUnconnectedOutLayersNames());
        
        blob_ = datastream_f2matc3_f(output_img[0], model_image_size);// [0-1]
    }
    qint64 elapsedTime = timer.nsecsElapsed(); // ��ȡ����Ϊ��λ�ĳ���ʱ��
    qDebug() << "-----Execution time: " << elapsedTime / 1e6 << " ms."; // ת����ɺ��벢���
    

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
    // // �����һ������ľ�ֵ�ͱ�׼��
    cv::Scalar meanValue(0.709, 0.381, 0.224);  // ��ֵ
    cv::Scalar stdDevValue(0.127, 0.079, 0.043); // ��׼��
    // // �Ե�����С���ͼ����й�һ��
    cv::Mat normalizedImage = normalizeImage(image, meanValue, stdDevValue);
    cv::Mat ret = one_image_inference(normalizedImage, model_image_size, isEngine);
    //cv::imshow("ret", ret);
    return ret;
}