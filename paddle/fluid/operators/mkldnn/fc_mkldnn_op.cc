/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,     
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <unordered_map>
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using namespace mkldnn;


template <typename T>
class FCMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<Tensor>("Out");
    auto w_dims = w->dims();
    auto out_dims = output->dims();
    LOG(INFO)<<"input dimention: "<<input->dims();
    LOG(INFO)<<"weight dimention: "<<w_dims;
    LOG(INFO)<<"output dimention: "<<out_dims;   
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& cpu_engine = dev_ctx.GetEngine();
    
    /* Create a vector to store the topology primitives */
    std::vector<primitive> net;
    LOG(INFO)<<"--------0---------"; 
    std::vector<int> fc_src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> fc_weights_tz =
        paddle::framework::vectorize2int(w->dims());
    std::vector<int> fc_bias_tz = paddle::framework::vectorize2int(bias->dims());
    std::vector<int> fc_dst_tz = paddle::framework::vectorize2int(output->dims());
   /* Set Scaling mode for int8 quantizing */
    const std::vector<float> src_scales = { 1.8f };
    const std::vector<float> weight_scales = { 2.0f };
    const std::vector<float> bias_scales = { 1.0f };
    const std::vector<float> dst_scales = { 0.55f };

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int fc_mask = 1; // 1 << output_channel_dim
    
    /* Allocate input and output buffers for user data */
    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();
    
    LOG(INFO)<<"---------1----------";
    /* create memory for user data */
    auto user_src_memory = memory(
            { { { fc_src_tz }, memory::data_type::f32, memory::format::ncw },
                    cpu_engine },
            to_void_cast<T>(input_data));
    auto user_weights_memory
            = memory({ { { fc_weights_tz }, memory::data_type::f32,
                               memory::format::nc },
                             cpu_engine },
            to_void_cast<T>(w_data));
    auto user_bias_memory = memory(
            { { { fc_bias_tz }, memory::data_type::f32, memory::format::x },
                    cpu_engine },
            to_void_cast<T>(bias_data));
     
    LOG(INFO)<<"---------2----------";
    /* create memory descriptors for fc data w/ no specified format */
    auto fc_src_md = memory::desc(
            { fc_src_tz }, memory::data_type::s8, memory::format::any);
    auto fc_bias_md = memory::desc(
            { fc_bias_tz }, memory::data_type::s8, memory::format::any);
    auto fc_weights_md = memory::desc(
            { fc_weights_tz }, memory::data_type::s8, memory::format::any);
    auto fc_dst_md = memory::desc(
            { fc_dst_tz }, memory::data_type::s8, memory::format::any);

    LOG(INFO)<<"---------3----------";
    /* create a inner product */
    auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md);

    LOG(INFO)<<"---------4----------";
    /* define the fc attributes */
    primitive_attr fc_attr;
    fc_attr.set_int_output_round_mode(round_mode::round_nearest);
    fc_attr.set_output_scales(fc_mask, src_scales);

    LOG(INFO)<<"---------5----------";
    /* check if int8 inner product is supported */
    try {
        auto fc_prim_desc = inner_product_forward::primitive_desc(
                fc_desc, fc_attr, cpu_engine);
    } catch (error &e) {
        if (e.status == mkldnn_unimplemented) {
            std::cerr << "AVX512-BW support or Intel(R) MKL dependency is "
            "required for int8 inner product" << std::endl;
        }
        throw;
    }
   
    auto fc_prim_desc = inner_product_forward::primitive_desc(
            fc_desc, fc_attr, cpu_engine);

    LOG(INFO)<<"---------6----------";
    /* Next: create memory primitives for the inner product's input data
     * and use reorder to quantize the values into int8 */
    auto fc_src_memory = memory(fc_prim_desc.src_primitive_desc());
    primitive_attr src_attr;
    src_attr.set_int_output_round_mode(round_mode::round_nearest);
    src_attr.set_output_scales(src_mask, src_scales);
    auto src_reorder_pd
            = reorder::primitive_desc(user_src_memory.get_primitive_desc(),
                    fc_src_memory.get_primitive_desc(), src_attr);
    net.push_back(reorder(src_reorder_pd, user_src_memory, fc_src_memory));

    LOG(INFO)<<"---------7----------";
    auto fc_weights_memory = memory(fc_prim_desc.weights_primitive_desc());
    primitive_attr weight_attr;
    weight_attr.set_int_output_round_mode(round_mode::round_nearest);
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd
            = reorder::primitive_desc(user_weights_memory.get_primitive_desc(),
                    fc_weights_memory.get_primitive_desc(), weight_attr);
    net.push_back(reorder(
            weight_reorder_pd, user_weights_memory, fc_weights_memory));

    LOG(INFO)<<"---------8----------";
    auto fc_bias_memory = memory(fc_prim_desc.bias_primitive_desc());
    primitive_attr bias_attr;
    bias_attr.set_int_output_round_mode(round_mode::round_nearest);
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd
            = reorder::primitive_desc(user_bias_memory.get_primitive_desc(),
                    fc_bias_memory.get_primitive_desc(), bias_attr);
    net.push_back(reorder(bias_reorder_pd, user_bias_memory, fc_bias_memory));

    auto fc_dst_memory = memory(fc_prim_desc.dst_primitive_desc());

    LOG(INFO)<<"---------9----------";
    /* create inner product primitive and add it to net */
    net.push_back(inner_product_forward(fc_prim_desc, fc_src_memory,
            fc_weights_memory, fc_bias_memory, fc_dst_memory));

    /* Convert data back into fp32 and compare values with u8.
      * Note: data is unsigned since there are no negative value
      * after ReLU */

    LOG(INFO)<<"---------10----------";
    const T* output_data = output->data<T>();
    /* Create a memory primitive for user data output */
    auto user_dst_memory = memory(
            { { { fc_dst_tz }, memory::data_type::f32, memory::format::nchw },
                    cpu_engine },
            to_void_cast<T>(output_data));

    primitive_attr dst_attr;
    dst_attr.set_int_output_round_mode(round_mode::round_nearest);
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = reorder::primitive_desc(fc_dst_memory.get_primitive_desc(),
                    user_dst_memory.get_primitive_desc(), dst_attr);
    
    LOG(INFO)<<"---------11----------";
    /* Convert the destination memory from inner product into user
 *      * data format if necessary */
    if (fc_dst_memory != user_dst_memory) {
        net.push_back(
                reorder(dst_reorder_pd, fc_dst_memory, user_dst_memory));
    }
   

      stream(stream::kind::eager).submit(net).wait();   
}
};

}
}

REGISTER_OP_KERNEL(fc, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNOpKernel<float>);
