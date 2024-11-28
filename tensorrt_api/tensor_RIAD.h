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
    struct infer_results<tensor_interface_type::tensor_RIAD> : public infer_results_base
    {
        cv::Mat reconstruct_image;
    }; 

    template<>
    struct input_params<tensor_interface_type::tensor_RIAD> : public input_params_base
    {
        std::string s = "这是RIAD的输入参数";
    };
    
    template<tensor_interface_type TensorType>
    class tensor_RIAD : public tensor_base<TensorType>
	{
        public: 
            constexpr static tensor_interface_type Type = TensorType;
  
            using Ptr = std::shared_ptr<tensor_RIAD<Type>>;

            tensor_RIAD(){};

            tensor_RIAD(std::string engine_file):tensor_base(engine_file){};

            void decode_output() override ;

    };

}