/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-01-19 09:09:57
 * @LastEditors: SJL
 * @LastEditTime: 2024-01-26 13:08:20
 */

#pragma once
#include "tensor_base.h"

namespace tenser_openex
{
    template<>
    struct infer_results<tensor_interface_type::tensor_classifier> : public infer_results_base
    {
        int class_id = 0;
        double class_confidence = 0.99;
    }; 

    template<>
    struct input_params<tensor_interface_type::tensor_classifier> : public input_params_base
    {
        std::string s = "这是分类的输入参数";
    };
        
    template<tensor_interface_type TensorType>
    class tensor_classifier : public tensor_base<TensorType>
	{
        public: 
            constexpr static tensor_interface_type Type = TensorType;
  
            using Ptr = std::shared_ptr<tensor_classifier<Type>>;

            tensor_classifier(){};

            tensor_classifier(std::string engine_file):tensor_base(engine_file){};

            void decode_output () override;

			//void get_results(std::vector<classifier_results>& results) ;

    };

}