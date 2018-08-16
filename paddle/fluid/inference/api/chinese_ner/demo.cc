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

/*
* This file contains a simple demo for how to take a model for inference.
*/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <thread>  //NOLINT
#include <vector>
#include <map>
#include <string.h>
#include <fstream>
#include <istream>
#include <thread>
#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <map>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "paddle/fluid/platform/enforce.h"
#include <algorithm>

using namespace std;
DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "Whether use gpu.");

namespace paddle {
	namespace demo {
		void Main(bool use_gpu) {
			//# 1. Create PaddlePredictor with a config.
			NativeConfig config;
			if (FLAGS_dirname.empty()) {
				LOG(INFO) << "Usage: ./chinese_ner --dirname=path/to/your/model";
				exit(1);
			}
			config.model_dir = FLAGS_dirname;
			config.use_gpu = use_gpu;
			config.fraction_of_gpu_memory = 0.15;
			config.device = 0;
			auto predictor =
				CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

			for (int batch_id = 0; batch_id < 1; batch_id++) {
				//# 2. Prepare input.
				std::string test_data("./data_file");
				std::ifstream fin;
				fin.open(test_data);
				if (!fin){
					std::cerr << "data file path is not correct" << std::endl;
				}
				std::string line;
				int64_t total_count = 0;
                                std::vector<std::string> line_vec;
                                std::vector<std::string> word_vec;
                                std::vector<std::string> target_vec;
                                std::vector<std::string> mention_vec;
				getline(fin, line);
                                boost::split(line_vec,line, boost::is_any_of(";"));
                                std::vector<std::string>::iterator it;
                                int i=0;
		                for (it = line_vec.begin(); it != line_vec.end(); ++it){
				    boost::split(line_vec, line, boost::is_any_of(";"));
				    if (i == 1){
                                        std::string word;
                                        word = *it;
                                        boost::split(word_vec, word, boost::is_any_of(" "));
                                    }
			            if (i == 2){
                                        std::string target;
                                        target = *it;
                                        boost::split(target_vec, target, boost::is_any_of(" "));
                                    }
				    if (i == 3){
                                        std::string mention;
                                        mention = *it;
                                        boost::split(mention_vec, mention, boost::is_any_of(" "));
                                    }
                                    i++;
				}
			
                                copy (word_vec.begin(), word_vec.end(), ostream_iterator<std::string> (cout, ", "));
                                cout<<" "<<endl;
                                copy (target_vec.begin(), target_vec.end(), ostream_iterator<std::string> (cout, ", "));
                                cout<<" "<<endl;
                                copy (mention_vec.begin(), mention_vec.end(), ostream_iterator<std::string> (cout, ", "));
                                cout<<" "<<endl;
				PaddleTensor tensor_word;
				tensor_word.shape = std::vector<int>({ 1 });
				tensor_word.data = PaddleBuf(static_cast<void*>(&word_vec), sizeof(word_vec));
				tensor_word.dtype = PaddleDType::INT64;

				PaddleTensor tensor_mention;
				tensor_mention.shape = std::vector<int>({ 1 });
				tensor_mention.data = PaddleBuf(static_cast<void*>(&mention_vec), sizeof(mention_vec));
				tensor_mention.dtype = PaddleDType::INT64;

				// For simplicity, we set all the slots with the same data.
				//std::vector<PaddleTensor> slots(tensor_word, tensor_mention);
				std::vector<PaddleTensor> slots(4, tensor_word);

				//# 3. Run
				std::vector<PaddleTensor> outputs;
				CHECK(predictor->Run(slots, &outputs));

				//# 4. Get output.
				PADDLE_ENFORCE(outputs.size(), 1UL);
				// Check the output buffer size and result of each tid.
				PADDLE_ENFORCE(outputs.front().data.length(), 33168UL);
				float result[5] = { 0.00129761, 0.00151112, 0.000423564, 0.00108815,
					0.000932706 };
				const size_t num_elements = outputs.front().data.length() / sizeof(float);
				// The outputs' buffers are in CPU memory.
				for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
					PADDLE_ENFORCE(static_cast<float*>(outputs.front().data.data())[i],
						result[i]);
				}
			}
		}

		void MainThreads(int num_threads, bool use_gpu) {
			// Multi-threads only support on CPU
			// 0. Create PaddlePredictor with a config.
			NativeConfig config;
			config.model_dir = FLAGS_dirname;
			config.use_gpu = use_gpu;
			config.fraction_of_gpu_memory = 0.15;
			config.device = 0;
			auto main_predictor =
				CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

			std::vector<std::thread> threads;
			for (int tid = 0; tid < num_threads; ++tid) {
				threads.emplace_back([&, tid]() {
					// 1. clone a predictor which shares the same parameters
					auto predictor = main_predictor->Clone();
					constexpr int num_batches = 1;
					for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
						// 2. Dummy Input Data
						std::string test_data("./data_file");
						std::ifstream fin;
						fin.open(test_data);
						if (!fin){
							std::cerr << "data file path is not correct" << std::endl;
						}
						std::string line;
						int64_t total_count = 0;
                                                std::vector<std::string> line_vec;
                                                std::vector<std::string> word_vec;
                                                std::vector<std::string> target_vec;
                                                std::vector<std::string> mention_vec;
                                                getline(fin, line);
                                                boost::split(line_vec,line, boost::is_any_of(";"));
                                                std::vector<std::string>::iterator it;
                                                int i=0;
                                                for (it = line_vec.begin(); it != line_vec.end(); ++it){
                                                    boost::split(line_vec, line, boost::is_any_of(";"));
                                                    if (i == 1){
                                                        std::string word;
                                                        word = *it;
                                                        boost::split(word_vec, word, boost::is_any_of(" "));
                                                    }
                                                    if (i == 2){
                                                        std::string target;
                                                        target = *it;
                                                        boost::split(target_vec, target, boost::is_any_of(" "));
                                                    }
                                                    if (i == 3){
                                                        std::string mention;
                                                        mention = *it;
                                                        boost::split(mention_vec, mention, boost::is_any_of(" "));
                                                    }           
                                                    i++;
                                                }

                                                

						PaddleTensor tensor_word;
						tensor_word.shape = std::vector<int>({ 1 });
						tensor_word.data = PaddleBuf(static_cast<void*>(&word_vec), sizeof(word_vec));
						tensor_word.dtype = PaddleDType::INT64;

						PaddleTensor tensor_mention;
						tensor_mention.shape = std::vector<int>({ 1 });
						tensor_mention.data = PaddleBuf(static_cast<void*>(&mention_vec), sizeof(mention_vec));
						tensor_mention.dtype = PaddleDType::INT64;

						std::vector<PaddleTensor> inputs(4, tensor_mention);
						std::vector<PaddleTensor> outputs;

						// 3. Run
						CHECK(predictor->Run(inputs, &outputs));

						// 4. Get output.
						PADDLE_ENFORCE(outputs.size(), 1UL);
						// Check the output buffer size and result of each tid.
						PADDLE_ENFORCE(outputs.front().data.length(), 33168UL);
						float result[5] = { 0.00129761, 0.00151112, 0.000423564, 0.00108815,
							0.000932706 };
						const size_t num_elements =
							outputs.front().data.length() / sizeof(float);
						// The outputs' buffers are in CPU memory.
						for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
							PADDLE_ENFORCE(static_cast<float*>(outputs.front().data.data())[i],
								result[i]);
						}
					}
				});
			}
			for (int i = 0; i < num_threads; ++i) {
				threads[i].join();
			}
		}

	}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
	google::ParseCommandLineFlags(&argc, &argv, true);
	paddle::demo::Main(false /* use_gpu*/);
	paddle::demo::MainThreads(1, false /* use_gpu*/);
	paddle::demo::MainThreads(4, false /* use_gpu*/);
	if (FLAGS_use_gpu) {
		paddle::demo::Main(true /*use_gpu*/);
		paddle::demo::MainThreads(1, true /*use_gpu*/);
		paddle::demo::MainThreads(4, true /*use_gpu*/);
	}
	return 0;
}
