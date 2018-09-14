// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/platform/profiler.h"
#include <thread>

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data path");
DEFINE_int32(batch_size, 10, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");

namespace paddle {
namespace inference {

struct DataRecord {
  std::vector<std::vector<int64_t>> word_data_all, mention_data_all;
  std::vector<std::vector<int64_t>> rnn_word_datas, rnn_mention_datas;
  std::vector<size_t> lod;  // two inputs have the same lod info.
  std::vector<size_t> batch_iters;
  int num_threads;
  size_t batch_iter{0};
  size_t batch_size{1};
  size_t num_samples;  // total number of samples
  DataRecord() : num_threads(0), num_samples(0){};
  DataRecord NextBatch(int tid = 0) {
    DataRecord data;
    batch_iter = batch_iters[tid];
    size_t batch_end = batch_iter + batch_size;
    
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= word_data_all.size()) {
      data.word_data_all.assign(word_data_all.begin() + batch_iter,
                                word_data_all.begin() + batch_end);
      data.mention_data_all.assign(mention_data_all.begin() + batch_iter,
                                   mention_data_all.begin() + batch_end);
      // Prepare LoDs
      data.lod.push_back(0);
      CHECK(!data.word_data_all.empty());
      CHECK(!data.mention_data_all.empty());
      CHECK_EQ(data.word_data_all.size(), data.mention_data_all.size());
      for (size_t j = 0; j < data.word_data_all.size(); j++) {
        data.rnn_word_datas.push_back(data.word_data_all[j]);
        data.rnn_mention_datas.push_back(data.mention_data_all[j]);
        // calculate lod
        data.lod.push_back(data.lod.back() + data.word_data_all[j].size());
      }
    }
    batch_iter += batch_size;
    return data;
  }
  

  void Load(const std::string &path, const int threads) {
    num_threads = threads;
    num_samples = 0;
    std::ifstream file(path);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ';', &data);
      // load word data
      std::vector<int64_t> word_data;
      split_to_int64(data[1], ' ', &word_data);
      // load mention data
      std::vector<int64_t> mention_data;
      split_to_int64(data[3], ' ', &mention_data);
      word_data_all.push_back(std::move(word_data));
      mention_data_all.push_back(std::move(mention_data));
    }
    num_samples = num_lines;
    batch_iters.resize(num_threads);
    std::fill(batch_iters.begin(), batch_iters.end(), 0);
  }
 
  void Reset(int tid = 0){
      PADDLE_ENFORCE_LT(tid,FLAGS_num_threads);
      batch_iters[tid] = 0;
  }
};

struct PredictStats {
  long total_samples;
  long total_iters;
  double total_time;
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size, int tid = 0) {
  PADDLE_ENFORCE_LT(tid, FLAGS_num_threads);
  data->batch_iter = data->batch_iters[tid];

  PaddleTensor lod_word_tensor, lod_mention_tensor;
  lod_word_tensor.name = "word";
  lod_mention_tensor.name = "mention";
  auto one_batch = data->NextBatch(tid);
  int size = one_batch.lod[one_batch.lod.size() - 1];  // token batch size
  lod_word_tensor.shape.assign({size, 1});
  lod_word_tensor.lod.assign({one_batch.lod});
  lod_mention_tensor.shape.assign({size, 1});
  lod_mention_tensor.lod.assign({one_batch.lod});
  // assign data
  TensorAssignData<int64_t>(&lod_word_tensor, one_batch.rnn_word_datas);
  TensorAssignData<int64_t>(&lod_mention_tensor, one_batch.rnn_mention_datas);
  // Set inputs.
  input_slots->assign({lod_word_tensor, lod_mention_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
  data->batch_iters[tid] += batch_size;
}

struct DataRecord reader;
std::vector<struct PredictStats> stats;
// the first inference result
const int chinese_ner_result_data[] = {30, 45, 41, 48, 17, 26,
                                       48, 39, 38, 16, 25};

void TestChineseNERPrediction(bool use_analysis) {
  reader.Load(FLAGS_infer_data, FLAGS_num_threads);
  stats.resize(FLAGS_num_threads);  

  NativeConfig config;
  config.prog_file = FLAGS_infer_model + "/__model__";
  config.param_file = FLAGS_infer_model + "/param";
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;

  std::vector<PaddleTensor> input_slots, outputs;
  std::unique_ptr<PaddlePredictor> predictor;
  Timer timer;
  if (use_analysis) {
    AnalysisConfig cfg;
    cfg.prog_file = FLAGS_infer_model + "/__model__";
    cfg.param_file = FLAGS_infer_model + "/param";
    cfg.use_gpu = false;
    cfg.device = 0;
    cfg.specify_input_name = true;
    cfg.enable_ir_optim = true;
    predictor =
        CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(cfg);
  } else {
    predictor =
        CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  }

  if (FLAGS_test_all_data) {
    LOG(INFO) << "test all data";
    double sum = 0;
    size_t num_samples;
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<PaddlePredictor>> predictors;
    for (int tid = 0; tid < FLAGS_num_threads; ++tid) {
       if (use_analysis) {
           AnalysisConfig cfg;
           cfg.prog_file = FLAGS_infer_model + "/__model__";
           cfg.param_file = FLAGS_infer_model + "/param";
           cfg.use_gpu = false;
           cfg.device = 0;
           cfg.specify_input_name = true;
           cfg.enable_ir_optim = true;
           predictors.emplace_back(
               CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
                   cfg));
        } else {

           predictors.emplace_back(
               CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config));
       }
    }
    for (int tid = 0; tid < FLAGS_num_threads; tid++) {
        int total_batches = 0;
        threads.emplace_back([&, tid]() {
        for (int i = 0; i < FLAGS_repeat; i++) {
             reader.Reset(tid);
             int num_batches = 0;
             num_samples = reader.num_samples;
             for (size_t bid = 0; bid < num_samples / FLAGS_batch_size; ++bid) {
                  PrepareInputs(&input_slots, &reader, FLAGS_batch_size, tid);
                  timer.tic();
                  predictors[tid]->Run(input_slots, &outputs);
                  sum += timer.toc();
                  ++num_batches;
              }
              total_batches += num_batches;
         }
         // save stat
         PredictStats& stat = stats[tid];
         stat.total_samples = total_batches * FLAGS_batch_size;
         stat.total_time = sum;
         stat.total_iters = total_batches;
       });
    }
    // Wait all threads done
    for (int i = 0; i < FLAGS_num_threads; i++) {
         threads[i].join();
         LOG(INFO)<<"wait all threads done."<< i;
    } 
    
    // collect statistic data
    long total_samples = std::accumulate(
        stats.begin(), stats.end(), 0,
        [](long a, PredictStats& b) { return a + b.total_samples; });
    long total_iters = std::accumulate(
        stats.begin(), stats.end(), 0,
        [](long a, PredictStats& b) { return a + b.total_iters; });
    double total_time = std::accumulate(
        stats.begin(), stats.end(), 0,
        [](double a, PredictStats& b) { return a + b.total_time; });
    
    LOG(INFO) << "==== Predict with all " << FLAGS_num_threads
              << " threads finished ====";
    LOG(INFO) << "Total samples: " << total_samples
              << ", Total time(ms): " << total_time;
    LOG(INFO) << "Total iterations: " << total_iters
              << ", BatchSize: " << FLAGS_batch_size;
    LOG(INFO) << "Total QPS: "
              << total_samples * 1000 / (total_time / FLAGS_num_threads)
              << ", Aver QPS per thread: " << total_samples * 1000 / total_time;
    LOG(INFO) << "Average latency per iter (ms): " << total_time / total_iters;
    LOG(INFO) << "Average latency per sample (ms): "
              << total_time / total_samples;
    /*
    LOG(INFO) << "total number of samples: " << num_samples;
    PrintTime(FLAGS_batch_size, FLAGS_repeat, 1, 0, sum / FLAGS_repeat);
    LOG(INFO) << "average latency of each sample: "
              << sum / FLAGS_repeat / num_samples;*/
    return;
  }
  /*
  // Prepare inputs.
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  PrepareInputs(&input_slots, &data, FLAGS_batch_size);

  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    predictor->Run(input_slots, &outputs);
  }
  PrintTime(FLAGS_batch_size, FLAGS_repeat, 1, 0, timer.toc() / FLAGS_repeat);

  PADDLE_ENFORCE(outputs.size(), 1UL);
  auto &out = outputs[0];
  size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                [](int a, int b) { return a * b; });
  PADDLE_ENFORCE_GT(size, 0);
  int64_t *result = static_cast<int64_t *>(out.data.data());
  for (size_t i = 0; i < std::min(11UL, size); i++) {
    PADDLE_ENFORCE(result[i], chinese_ner_result_data[i]);
  }

  if (use_analysis) {
    // run once for comparion as reference
    auto ref_predictor =
        CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
    std::vector<PaddleTensor> ref_outputs_slots;
    ref_predictor->Run(input_slots, &ref_outputs_slots);
    EXPECT_EQ(ref_outputs_slots.size(), outputs.size());
    auto &ref_out = ref_outputs_slots[0];
    size_t ref_size =
        std::accumulate(ref_out.shape.begin(), ref_out.shape.end(), 1,
                        [](int a, int b) { return a * b; });
    EXPECT_EQ(size, ref_size);
    int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(pdata_ref[i], result[i]);
    }

    AnalysisPredictor *analysis_predictor =
        dynamic_cast<AnalysisPredictor *>(predictor.get());
    auto &fuse_statis = analysis_predictor->analysis_argument()
                            .Get<std::unordered_map<std::string, int>>(
                                framework::ir::kFuseStatisAttr);
    for (auto &item : fuse_statis) {
      LOG(INFO) << "fused " << item.first << " " << item.second;
    }
    int num_ops = 0;
    for (auto &node :
         analysis_predictor->analysis_argument().main_dfg->nodes.nodes()) {
      if (node->IsFunction()) {
        ++num_ops;
      }
    }
    LOG(INFO) << "has num ops: " << num_ops;
    ASSERT_TRUE(fuse_statis.count("fc_fuse"));
    ASSERT_TRUE(fuse_statis.count("fc_gru_fuse"));
    EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
    EXPECT_EQ(fuse_statis.at("fc_gru_fuse"), 2);
    EXPECT_EQ(num_ops, 14);
  }*/
}

TEST(Analyzer_Chinese_ner, native) { TestChineseNERPrediction(false); }

TEST(Analyzer_Chinese_ner, analysis) { TestChineseNERPrediction(true); }

}  // namespace inference
}  // namespace paddle
