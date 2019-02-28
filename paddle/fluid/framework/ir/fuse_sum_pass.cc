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

#include "paddle/fluid/framework/ir/fuse_sum_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FuseSumPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fused_sum", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* y = gpd.mutable_pattern()
                ->NewNode("fused_sum/y")
                ->AsInput()
                ->assert_is_op_input("softsign", "X");
  patterns::FuseSum fuse_sum_pattern(gpd.mutable_pattern(), "fused_sum");
  fuse_sum_pattern(y);

  int found_fp_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Fuse Pyramid fuse";
    GET_IR_NODE_FROM_SUBGRAPH(fuse_sum1, fuse_sum1, fuse_sum_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_sum2, fuse_sum2, fuse_sum_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(emb11, emb11, fuse_sum_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(emb12, emb12, fuse_sum_pattern);
    //GET_IR_NODE_FROM_SUBGRAPH(emb21, emb21, fuse_sum_pattern);
    //GET_IR_NODE_FROM_SUBGRAPH(emb22, emb22, fuse_sum_pattern);
  
    GET_IR_NODE_FROM_SUBGRAPH(fuse_sum1_input, fuse_sum1_input, fuse_sum_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_sum2_input, fuse_sum2_input, fuse_sum_pattern);


    // Create an FC Node.
    OpDesc desc;
    std::string fp_x1_in = fuse_sum1_input->Name();
    std::string fp_x2_in = fuse_sum2_input->Name();

    std::string fp_emb1_in = emb11->Name();
    std::string fp_emb2_in = emb12->Name();
    std::string fp_out = subgraph.at(y)->Name();

    //mod_by.push_back(boost::get<int>(hash3->Op()->GetAttr("mod_by")));


    desc.SetInput("X0", std::vector<std::string>({fp_x1_in}));
    desc.SetInput("X1", std::vector<std::string>({fp_x2_in}));
    desc.SetInput("W0", std::vector<std::string>({fp_emb1_in}));
    desc.SetInput("W1", std::vector<std::string>({fp_emb2_in}));
    desc.SetOutput("Out", std::vector<std::string>({fp_out}));

    std::vector<int> win_size;
    std::vector<int> mod_by;
    std::vector<int> num_hash;

    win_size = boost::get<std::vector<int>>(fuse_sum1->Op()->GetAttr("win_size"));

    mod_by = boost::get<std::vector<int>>(fuse_sum1->Op()->GetAttr("mod_by"));

    num_hash = boost::get<std::vector<int>>(fuse_sum1->Op()->GetAttr("num_hash"));

    desc.SetAttr("num_hash", num_hash);
    desc.SetAttr("mod_by", mod_by);
    desc.SetAttr("win_size", win_size);
    desc.SetAttr("pad_value", fuse_sum1->Op()->GetAttr("pad_value"));
    
    desc.SetType("fused_enum_hash_emd_pool");
    auto fp_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {fuse_sum1, fuse_sum2});

    PADDLE_ENFORCE(subgraph.count(y));
    IR_NODE_LINK_TO(fuse_sum1_input, fp_node);
    IR_NODE_LINK_TO(fuse_sum2_input, fp_node);
    IR_NODE_LINK_TO(emb11, fp_node);
    IR_NODE_LINK_TO(emb12, fp_node);
    IR_NODE_LINK_TO(fp_node, subgraph.at(y));

    found_fp_count++;
  };


  gpd(graph.get(), handler);

  AddStatis(found_fp_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_sum_pass, paddle::framework::ir::FuseSumPass);
