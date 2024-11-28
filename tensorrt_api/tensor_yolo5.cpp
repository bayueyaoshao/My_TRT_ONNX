/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-23 09:48:17
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 13:05:45
 */
#pragma once
#include "tensor_yolo5.h"
namespace tenser_openex
{
    void tensor_yolo5<tensor_interface_type::tensor_yolo5>::decode_output()
    {
        for(size_t batch_idx = 0; batch_idx < params_.batch_size; batch_idx++)
        {   
            infer_results<tensor_interface_type::tensor_yolo5> result;

            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            std::vector<int> class_ids;
            auto dets = this->output_size_ / (params_.class_nums + 5);

            float x_factor = 1/this->scale_;
            float y_factor = 1/this->scale_;
            for (int j = 0; j < dets; ++j)
            {
                float confidence = *(batch_idx * dets * (params_.class_nums + 5) + this->output_data_host_ + j * (params_.class_nums + 5) + 4);
                if (confidence >= 0.5)
                {
                    float* classes_scores = this->output_data_host_ + j * (params_.class_nums + 5) + 5;
                    cv::Mat scores(1, params_.class_nums, CV_32FC1, classes_scores);
                    cv::Point class_id;
                    double max_class_scores;
                    minMaxLoc(scores, 0, &max_class_scores, 0, &class_id);
                    if (max_class_scores > 0.5)
                    {
                        float cx = *(this->output_data_host_ + j * (params_.class_nums + 5) + 0);
                        float cy = *(this->output_data_host_ + j * (params_.class_nums + 5) + 1);
                        float w = *(this->output_data_host_ + j * (params_.class_nums + 5) + 2);
                        float h = *(this->output_data_host_ + j * (params_.class_nums + 5) + 3);
                        int left = int((cx - 0.5 * w) * x_factor);
                        int top = int((cy - 0.5 * h) * y_factor);
                        int width = int(w * x_factor);
                        int height = int(h * y_factor);
                        if ((left > 0) && (top > 0) && (width > 0) && (height > 0))
                        {
                            boxes.push_back(cv::Rect(left, top, width, height));
                            confidences.push_back(confidence);
                            class_ids.push_back(class_id.x);
                        }
                        
                    }
                }

            }




            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, float(0.25), float(0.45), indices);
            cv::Mat img = this->src_img_vec_[batch_idx];
            if (indices.size() > 0)
            {
                for (int i = 0; i < indices.size(); i++)
                {
                    int idx = indices[i];
                    cv::Rect box = boxes[idx];
                    int left = box.x;
                    int top = box.y;
                    int width = box.width;
                    int height = box.height;

                    result.confidences.push_back(confidences[idx]);
                    result.boxes.push_back(boxes[idx]);
                    result.class_ids.push_back(class_ids[idx]);

                  // cv::rectangle(img, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 2);

                }
            }
                                                
            //cv::imshow("1",img);
            //cv::waitKey(0);
            results_.push_back(result);

        }         
    }

}