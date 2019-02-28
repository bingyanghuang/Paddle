/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
struct EmbeddingVSumFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t table_height = table_t->dims()[0];
    int64_t table_width = table_t->dims()[1];
    int64_t out_width = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    int64_t idx_width = ids_t->numel() / ids_lod.back();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_LE(table_width * idx_width, out_width);
    PADDLE_ENFORCE_GT(ids_lod.size(), 1UL);

    jit::emb_seq_pool_attr_t attr(table_height, table_width, 0, idx_width,
                                  out_width, jit::SeqPoolType::kSum);
    for (size_t i = 0; i != ids_lod.size() - 1; ++i) {
      attr.index_height = ids_lod[i + 1] - ids_lod[i];
      auto emb_seqpool = jit::Get<jit::kEmbSeqPool, jit::EmbSeqPoolTuples<T>,
                                  platform::CPUPlace>(attr);
      emb_seqpool(table, ids + ids_lod[i] * idx_width, output + i * out_width,
                  &attr);
    }
  }
};

template <typename DeviceContext, typename T>
class FusedEnumHashEmdPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    for (auto input_name : input_names) {
      auto *in = context.Input<LoDTensor>(input_name);
      auto win_size = context.Attr<std::vector<int>>("win_size");
      int pad_value = context.Attr<int>("pad_value");

      auto in_dims = in->dims();
      auto in_lod = in->lod();

      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
          "The actual input data's size mismatched with LoD information.");

      // enumerate
      int max_win_size = *std::max_element(win_size.begin(), win_size.end());
      LoDTensor enum_out;
      enum_out.Resize({in_dims[0], max_win_size});

      // Generate enumerate sequence set
      auto lod0 = in_lod[0];
      auto in_data = in->data<int64_t>();
      auto enum_out_data = enum_out.mutable_data<int64_t>(context.GetPlace());

      for (size_t i = 0; i < lod0.size() - 1; ++i) {
        for (size_t idx = lod0[i]; idx < lod0[i + 1]; ++idx) {
          for (int word_idx = 0; word_idx < max_win_size; ++word_idx) {
            size_t word_pos = idx + word_idx;
            enum_out_data[max_win_size * idx + word_idx] =
                word_pos < lod0[i + 1] ? in_data[word_pos] : pad_value;
          }
        }
      }

      // hash
      std::vector<int> num_hash = context.Attr<std::vector<int>>("num_hash");
      std::vector<int> mod_by = context.Attr<std::vector<int>>("mod_by");
      PADDLE_ENFORCE(num_hash.size() == mod_by.size() &&
                         num_hash.size() == win_size.size(),
                     "All attributes's count should be equal!");
      auto seq_length = in_dims[0];
      for (auto hash_len : num_hash) {
        LoDTensor lod_tensor;
        lod_tensor.set_lod(in_lod);
        lod_tensor.Resize({seq_length, hash_len, 1});
        hash_out.push_back(std::move(lod_tensor));
      }
      for (int win_idx = 0; win_idx < win_size.size(); win_idx++) {
        auto hash_out_data =
            hash_out[win_idx].mutable_data<int64_t>(context.GetPlace());
        auto *input = enum_out_data;
        auto last_dim = win_size[win_idx];
        for (int idx = 0; idx < seq_length; ++idx) {
          for (int ihash = 0; ihash != num_hash[win_idx]; ++ihash) {
            hash_out_data[idx * num_hash[win_idx] + ihash] =
                XXH64(input, sizeof(int) * last_dim, ihash) % mod_by[win_idx];
          }
          input += max_win_size;
        }
      }

      // seq_emd_pool
      EmbeddingVSumFunctor<T> functor;
      const LoDTensor *table1_var = context.Input<LoDTensor>("W1");
      auto table1_tz = paddle::framework::vectorize2int(table1_var->dims());
      for (int idx = 0; idx < num_hash.size(); idx++) {
        LoDTensor lod_tensor;
        lod_tensor.set_lod(in_lod);
        lod_tensor.Resize({static_cast<int>(lod0.size() - 1),
                           num_hash[idx] * table1_tz.back()});
        lod_tensor.mutable_data<T>(context.GetPlace());
        functor(context, table1_var, &hash_out[idx], &lod_tensor);
        seq_pool_out.push_back(std::move(lod_tensor));
      }

      {
        const LoDTensor *table0_var = context.Input<LoDTensor>("W0");
        auto table0_tz = paddle::framework::vectorize2int(table0_var->dims());

        LoDTensor lod_tensor;
        lod_tensor.set_lod(in_lod);
        lod_tensor.Resize(
            {static_cast<int>(lod0.size() - 1), table0_tz.back()});
        lod_tensor.mutable_data<T>(context.GetPlace());
        functor(context, table0_var, in, &lod_tensor);
        seq_pool_out.push_back(lod_tensor);
      }

      // Clear resource
      hash_out.clear();
    }

    // sum
    bool in_place = false;
    auto &in_vars = seq_pool_out;
    size_t in_num = in_vars.size();

    auto *out = context.Output<LoDTensor>("Out");
    if (!in_place) {
      out->mutable_data<T>(context.GetPlace());
    }
    auto result = EigenVector<T>::Flatten(*out);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();
    int start = in_place ? 1 : 0;
    if (!in_place) {
      if (in_num >= 2) {
        auto &in_0 = in_vars[0];
        auto &in_1 = in_vars[1];
        if (in_0.numel() && in_1.numel()) {
          auto in_0_e = EigenVector<T>::Flatten(in_0);
          auto in_1_e = EigenVector<T>::Flatten(in_1);
          result.device(place) = in_0_e + in_1_e;
          start = 2;
        }
      }
      if (start != 2) {
        math::SetConstant<DeviceContext, T> constant_functor;
        constant_functor(context.template device_context<DeviceContext>(), out,
                         static_cast<T>(0));
      }
    }

    // If in_place, just skip the first tensor
    for (size_t i = start; i < in_num; i++) {
      {
        auto &in_t = in_vars[i];
        if (in_t.numel() == 0) {
          continue;
        }
        auto in = EigenVector<T>::Flatten(in_t);
        result.device(place) = result + in;
      }
    }

    // Clear resource
    seq_pool_out.clear();
  }

 private:
  mutable std::vector<LoDTensor> hash_out;
  mutable std::vector<LoDTensor> seq_pool_out;
  const std::vector<std::string> input_names = {"X0", "X1"};
};

}  // namespace operators
}  // namespace paddle
