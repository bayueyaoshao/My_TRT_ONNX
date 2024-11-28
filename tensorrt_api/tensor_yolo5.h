/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-23 09:47:07
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 09:45:40
 */

#pragma once
#include "tensor_base.h"

namespace tenser_openex
{

    template<>
    struct infer_results<tensor_interface_type::tensor_yolo5> : public infer_results_base
    {
        std::vector<int> class_ids;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
    }; 

    template<>
    struct input_params<tensor_interface_type::tensor_yolo5> : public input_params_base
    {
        std::string s = "这是yolo5的输入参数";
    };
    
    template<tensor_interface_type TensorType>
    class tensor_yolo5 : public tensor_base<TensorType>
	{
        public: 
            constexpr static tensor_interface_type Type = TensorType;
  
            using Ptr = std::shared_ptr<tensor_yolo5<Type>>;

            tensor_yolo5(){};

            tensor_yolo5(std::string engine_file):tensor_base(engine_file){};

            void decode_output() override ;

    };

}