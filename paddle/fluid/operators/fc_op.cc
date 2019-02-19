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

#include "paddle/fluid/operators/fc_op.h"
#include <vector>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "mkldnn.hpp"
#include <chrono>


namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

using namespace mkldnn;

void FCOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "X(Input) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Out(Output) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("W"),
                 "W(Input) of Fully Connected should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");

  if (ctx->HasInput("Bias")) {
    auto bias_dims = ctx->GetInputDim("Bias");
    if (bias_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(bias_dims[0], 1, "The shape of Bias must be [1, dim].");
      PADDLE_ENFORCE_EQ(bias_dims[1], w_dims[1],
                        "The shape of Bias must be [1, dim].");
    } else if (bias_dims.size() == 1) {
      PADDLE_ENFORCE_EQ(bias_dims[0], w_dims[1],
                        "The shape of Bias must be [1, dim].");
    }
  }

//  if (ctx->Attrs().Get<bool>("use_mkldnn")) {
//    PADDLE_ENFORCE(in_dims.size() == 2 || in_dims.size() == 4,
//                   "Fully Connected input should be 2-D or 4-D tensor.");
//  }
  PADDLE_ENFORCE_EQ(w_dims.size(), 2UL,
                    "Fully Connected input should be 2-D tensor.");
  int in_num_col_dims = ctx->Attrs().Get<int>("in_num_col_dims");
  PADDLE_ENFORCE_GT(
      in_dims.size(), in_num_col_dims,
      "The input tensor Input's rank of FCOp should be larger than "
      "in_num_col_dims.");

  auto in_mat_dims = framework::flatten_to_2d(in_dims, in_num_col_dims);
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1], w_dims[0],
      "Fully Connected input and weigth size do not match. %s, %s");

  std::vector<int64_t> output_dims;
  output_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    output_dims.push_back(in_dims[i]);
  }
  output_dims.push_back(w_dims[1]);

  ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
  ctx->ShareLoD("Input", "Out");
}

framework::OpKernelType FCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  if (ctx.Attr<bool>("use_mkldnn")) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
  return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                 ctx.GetPlace(), layout, library);
}

void FCOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");

  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), w_dims);
  }

  if (ctx->HasInput("Bias")) {
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
                   "Should have bias grad");
    auto bias_dims = ctx->GetInputDim("Bias");
    ctx->SetOutputDim(framework::GradVarName("Bias"), bias_dims);
  }
}

framework::OpKernelType FCOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  if (ctx.Attr<bool>("use_mkldnn")) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
  return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                 ctx.GetPlace(), layout, library);
}

void FCOpMaker::Make() {
  AddInput("Input", "(Tensor), The input tensor of fully connected operator.");
  AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
  AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
      .AsDispensable();
  AddAttr<int>("in_num_col_dims",
               "(int, default 1), The fc op can take tensors with more than "
               "two dimensions as its inputs.")
      .SetDefault(1)
      .EqualGreaterThan(1);
  AddOutput("Out", "(Tensor) The output tensor of fully connected operator. ");
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddComment(R"DOC(
  Fully Connected Operator.

  The fully connected operation calculates the output based on the input, weights and bias.
  The size of each dimension of the parameters checked in the infer-shape.
)DOC");
}

template <typename T>
class FCOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    //auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<Tensor>("Out");
    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto out_dims = output->dims();
    int M = framework::product(out_dims) / out_dims[out_dims.size() - 1];
    int N = w_dims[1];
    int K = w_dims[0];
    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    framework::Tensor A;
    framework::Tensor B;
    framework::Tensor C;
    const T* A_data = A.mutable_data<T>(input_dims, platform::CPUPlace());
    const T* B_data = B.mutable_data<T>(w_dims, platform::CPUPlace());
    T* C_data = C.mutable_data<T>(out_dims, platform::CPUPlace());
    
    auto start1 = std::chrono::steady_clock::now();
    //Find the max value of inputs and weights
    auto input_2d = input_dims.size() > 2 
                    ? framework::flatten_to_2d(input_dims, input_dims.size()-1)
                    : input_dims;

    float max_input = 0;
    float max_weight =0;
    for (int i = 0; i < input_2d[0]; i++ ){
        for(int j = 0; j < input_2d[1]; j++){
            max_input = std::max(std::abs(input_data[i*input_2d[0]+j]), max_input);
        }
    }
            
    for (int m = 0; m < w_dims[0]; m++){
        for (int n = 0; n < w_dims[1]; n++) {
            max_weight = std::max(std::abs(w_data[ m * w_dims[0] + n]), max_weight);
        }
    }   
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds> 
                            (std::chrono::steady_clock::now() - start1);

     auto start2 = std::chrono::steady_clock::now();
    //Quantize inputs and weights by calling mkldnn reorder primitive
    auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    std::vector<primitive> pipeline;
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(input->dims());

    mkldnn::primitive_attr attri;
    int mask = 0;
    attri.set_output_scales(mask, {255/max_input});
    auto src_md = platform::MKLDNNMemDesc({src_tz[0], src_tz[1], src_tz[2]}, memory::data_type::f32,
                                          memory::format::ncw);
    auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);
    auto src_memory =
        std::make_shared<mkldnn::memory>(src_pd, to_void_cast<T>(input_data));

    std::shared_ptr<primitive::at> src_memory_p =
        std::shared_ptr<primitive::at>(new primitive::at(*src_memory));
    std::shared_ptr<mkldnn::memory::primitive_desc> dst_pd;
    std::shared_ptr<mkldnn::memory> dst_memory;
    auto dst_md = platform::MKLDNNMemDesc(
        {dst_tz[0],dst_tz[1],dst_tz[2]}, memory::data_type::u8,
        memory::format::ncw);
    ///////////
    dst_pd.reset(new mkldnn::memory::primitive_desc(dst_md, engine));
    dst_memory.reset(new mkldnn::memory(*dst_pd, to_void_cast<T>(A_data)));
    /////////// 
    auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(src_pd, *dst_pd, attri));
    auto reorder_p = std::shared_ptr<reorder>(
        new reorder(*reorder_pd, *src_memory_p, *dst_memory));
    pipeline.push_back(*reorder_p);
    
    std::vector<int> w_src_tz = paddle::framework::vectorize2int(w->dims());
    std::vector<int> w_dst_tz = paddle::framework::vectorize2int(w->dims());
    mkldnn::primitive_attr w_attri;
    w_attri.set_output_scales(mask, {127/max_weight});
    auto w_src_md = platform::MKLDNNMemDesc({w_src_tz[0], w_src_tz[1]}, memory::data_type::f32,
                                          memory::format::nc);
    auto w_src_pd = mkldnn::memory::primitive_desc(w_src_md, engine);
    auto w_src_memory =
        std::make_shared<mkldnn::memory>(w_src_pd, to_void_cast<T>(w_data));
    std::shared_ptr<primitive::at> w_src_memory_p =
        std::shared_ptr<primitive::at>(new primitive::at(*w_src_memory));
    std::shared_ptr<mkldnn::memory::primitive_desc> w_dst_pd;
    std::shared_ptr<mkldnn::memory> w_dst_memory;
    auto w_dst_md = platform::MKLDNNMemDesc(
        {w_dst_tz[0], w_dst_tz[1]}, memory::data_type::s8,
        mkldnn::memory::format::nc);
    w_dst_pd.reset(new mkldnn::memory::primitive_desc(w_dst_md, engine));
    w_dst_memory.reset(new mkldnn::memory(*w_dst_pd, to_void_cast<T>(B_data)));
    auto w_reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(w_src_pd, *w_dst_pd, w_attri));
    auto w_reorder_p = std::shared_ptr<reorder>(
        new reorder(*w_reorder_pd, *w_src_memory_p, *w_dst_memory));
    pipeline.push_back(*w_reorder_p);
    stream(stream::kind::eager).submit(pipeline).wait();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds> 
                            (std::chrono::steady_clock::now() - start2);

    auto start3 = std::chrono::steady_clock::now();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    math::FCCompute<platform::CPUDeviceContext, T>(
        blas, M, N, K, A_data, B_data, C_data, NULL);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds> 
                            (std::chrono::steady_clock::now() - start3);
                
    auto start4 = std::chrono::steady_clock::now();
    //Dequantize the output from int32 to fp32
    std::vector<primitive> pipeline2;
    std::vector<int> out_src_tz = paddle::framework::vectorize2int(out_dims);
    std::vector<int> out_dst_tz = paddle::framework::vectorize2int(out_dims);

    mkldnn::primitive_attr out_attri;
    out_attri.set_output_scales(mask, {1/(max_input*max_weight)});
    auto out_src_md = platform::MKLDNNMemDesc({out_src_tz[0], out_src_tz[1],out_src_tz[2] }, memory::data_type::s32,
                                          mkldnn::memory::format::ncw);
    auto out_src_pd = mkldnn::memory::primitive_desc(out_src_md, engine);
    auto out_src_memory =
        std::make_shared<mkldnn::memory>(out_src_pd, to_void_cast<T>(C_data));
    std::shared_ptr<primitive::at> out_src_memory_p =
        std::shared_ptr<primitive::at>(new primitive::at(*out_src_memory));
    std::shared_ptr<mkldnn::memory::primitive_desc> out_dst_pd;
    std::shared_ptr<mkldnn::memory> out_dst_memory;
    auto out_dst_md = platform::MKLDNNMemDesc(
        {out_dst_tz[0],out_dst_tz[1],out_dst_tz[2]}, memory::data_type::f32,
         mkldnn::memory::format::ncw);
    out_dst_pd.reset(new mkldnn::memory::primitive_desc(out_dst_md, engine));
    out_dst_memory.reset(new mkldnn::memory(*out_dst_pd, to_void_cast<T>(output_data)));
    auto out_reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(out_src_pd, *out_dst_pd, out_attri));
    auto out_reorder_p = std::shared_ptr<reorder>(
        new reorder(*out_reorder_pd, *out_src_memory_p, *out_dst_memory));
    pipeline2.push_back(*out_reorder_p);
    stream(stream::kind::eager).submit(pipeline2).wait();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds> 
                            (std::chrono::steady_clock::now() - start4);

    LOG(INFO)<<"Calucate the max value time: "<<duration1.count();
    LOG(INFO)<<"Reorder input and weights time: "<<duration2.count();
    LOG(INFO)<<"FC MKL s8u8 compute time: "<<duration3.count();    
    LOG(INFO)<<"Reorder output time: "<<duration4.count();
    // TODO(TJ): fuse act
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fc, ops::FCOp, ops::FCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fc_grad, ops::FCOpGrad);
REGISTER_OP_CPU_KERNEL(fc, ops::FCOpKernel<float>);
