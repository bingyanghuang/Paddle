//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/fused/fused_enum_hash_emd_pool_op.h"
#include <map>

namespace paddle {
namespace operators {

class FusedEnumHashEmdPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("W0"),
        "Input(W1) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("W1"),
        "Input(W2) of FusedEnumHashEmdPool operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out1) of FusedEnumHashEmdPool operator should not be null.");

    const auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dims.size(), 2,
        "Input(X) of FusedEnumHashEmdPool operator's rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[1], 1,
                      "Input(X) of FusedEnumHashEmdPool operator's 2nd "
                      "dimension should be 1.");

    std::string input_name = "X";
    std::string output_name = "Out";

    auto table0_dims = ctx->GetInputDim("W0");
    PADDLE_ENFORCE_EQ(table0_dims.size(), 2);

    int64_t last_dim = table0_dims[1];
    auto table1_dims = ctx->GetInputDim("W1");
    PADDLE_ENFORCE_EQ(table1_dims.size(), 2);

    const auto num_hash = ctx->Attrs().Get<std::vector<int>>("num_hash");
    int max_num_hash = *std::max_element(num_hash.begin(), num_hash.end());

    PADDLE_ENFORCE_EQ(table0_dims[1], table1_dims[1] * max_num_hash);

    if (ctx->IsRuntime()) {
      framework::Variable* ids_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs(input_name)[0]);
      const auto& ids_lod = ids_var->Get<LoDTensor>().lod();

      // in run time, the LoD of ids must be 1
      PADDLE_ENFORCE(ids_lod.size(), 1u,
                     "The LoD level of Input(Ids) must be 1");
      PADDLE_ENFORCE_GE(ids_lod[0].size(), 1u, "The LoD could NOT be empty");

      int64_t batch_size = ids_lod[0].size() - 1;

      // in run time, the shape from Ids -> output
      // should be [seq_length, 1] -> [batch_size, embedding_size]
      ctx->SetOutputDim(output_name,
                        framework::make_ddim({batch_size, last_dim}));
    } else {
      // in compile time, the lod level of ids must be 1
      framework::VarDesc* ids_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs(input_name)[0]);
      PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1);

      // in compile time, the shape from Ids -> output
      // should be [-1, 1] -> [-1, embedding_size]
      ctx->SetOutputDim(output_name, framework::make_ddim({-1, last_dim}));
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W0"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedEnumHashEmdPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(2-D LoDTensor with the 2nd dimension equal to 1) "
             "Input LoDTensor of SequenceEnumerate operator.");
    AddInput("W0",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("W1",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddOutput("Out", "(Tensor) The output tensor of sum operator.");
    AddAttr<std::vector<int>>("win_size",
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("pad_value", "(int) The enumerate sequence padding value.")
        .SetDefault(0);
    AddAttr<std::vector<int>>("num_hash", "")
        .SetDefault(std::vector<int>{
            1,
        });
    AddAttr<std::vector<int>>("mod_by", "")
        .SetDefault(std::vector<int>{
            100000,
        });
    AddComment(R"DOC(
Fused_enum_hash_emd_pool Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_enum_hash_emd_pool,
                             ops::FusedEnumHashEmdPoolOp,
                             ops::FusedEnumHashEmdPoolOpMaker);
REGISTER_OP_CPU_KERNEL(
    fused_enum_hash_emd_pool,
    ops::FusedEnumHashEmdPoolKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FusedEnumHashEmdPoolKernel<paddle::platform::CPUDeviceContext,
                                    double>);
