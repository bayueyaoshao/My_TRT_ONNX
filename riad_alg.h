#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<random>

void load_model(std::string onnx_path);
void loadEngineModel(std::string engine_path, int image_size);
cv::Mat normalizeImage(const cv::Mat& inputImage, const cv::Scalar& mean, const cv::Scalar& stdDev);

cv::Mat datastream_f2matc3_f(const cv::Mat& datastream, int image_size);
cv::Mat denormalization(const cv::Mat& normalizedImage);
cv::Mat one_image_inference(cv::Mat normalizedImage, int model_image_size, bool isEngine);

cv::Mat model_inference(std::string image_path, int model_image_size, cv::Mat& recMat, bool isEngine);

class riad_alg
{
};

