/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-23 09:48:17
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 13:05:45
 */
#pragma once
#include "tensor_RIAD.h"
#include<QDebug>

cv::Mat dsf2matc3_f(float* datastream, int image_size)
{
    int img_size = image_size * image_size * 1 * sizeof(float);
    cv::Mat blob_(image_size, image_size, CV_32FC1);
    std::memcpy(blob_.data, datastream, img_size);
    return blob_;
}


namespace tenser_openex
{
    void tensor_RIAD<tensor_interface_type::tensor_RIAD>::decode_output()
    {
        for(size_t batch_idx = 0; batch_idx < params_.batch_size; batch_idx++)
        {   
            infer_results<tensor_interface_type::tensor_RIAD> result;
            int output_size = this->output_size_;
            int calc_size = params_.input_height * params_.input_width * 1;
            cv::Mat blob_in = dsf2matc3_f(this->output_data_host_, params_.input_height);
            result.reconstruct_image = blob_in;
            results_.push_back(result);
        }         
    }

}
