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

#include "paddle/fluid/framework/ir/remove_sum_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> RmSumPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("remove_sum", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  PDNode* x;
  patterns::RmSum remove_sum_pattern(gpd.mutable_pattern(), "remove_sum");
  remove_sum_pattern(x);


  int found_fp_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Fuse Pyramid fuse";
    GET_IR_NODE_FROM_SUBGRAPH(sum, sum, remove_sum_pattern);
    GraphSafeRemoveNodes(graph.get(), {sum});

    found_fp_count++;
  };


  gpd(graph.get(), handler);

  AddStatis(found_fp_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_sum_pass, paddle::framework::ir::RmSumPass);
