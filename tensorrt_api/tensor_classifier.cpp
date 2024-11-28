/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-22 10:58:02
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 09:54:39
 */

#pragma once
#include "tensor_classifier.h"

namespace tenser_openex
{
    void tensor_classifier<tensor_interface_type::tensor_classifier>::decode_output()
    {
        for (size_t batch_idx = 0; batch_idx < params_.batch_size; batch_idx++)
        {
            cv::Mat scores(1, params_.class_nums, CV_32FC1, this->output_data_host_ + batch_idx*params_.class_nums);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            infer_results<tensor_interface_type::tensor_classifier> result;
            result.class_id = class_id.x;
            result.class_confidence = max_class_score;
            results_.push_back(result);
       }

    }

}