// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file eint8_outcept in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eint8_outpress or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/cpu_quantize_squash_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void CPUQuantizeSquashPass::SingleBranch(Graph* graph) const{
  GraphPatternDetector gpd;
  auto* int8_out = gpd.mutable_pattern()
                ->NewNode("squash_single/int8_out")
                ->AsInput()
                ->assert_is_op_input("dequantize", "Input");

  patterns::DequantQuantRM squash_pattern(gpd.mutable_pattern(),"squash_single");
  int case_ = 1;
  squash_pattern(int8_out, case_);

  int found_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle cpu quantize squash pass for dequantize->quantize";
    GET_IR_NODE_FROM_SUBGRAPH(dequant, dequantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant, quantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale = boost::get<float>(dequant->Op()->GetAttr("Scale"));
    float quant_scale = boost::get<float>(quant->Op()->GetAttr("Scale"));
    bool is_negative =
        boost::get<bool>(quant->Op()->GetAttr("is_negative_input"));

    if (dequant_scale == quant_scale){
       auto quant_out_var_name = quant_out->Name();
       auto next_op_inputs = next_op_desc->InputNames();
       for (auto name : next_op_inputs) {
           auto var_name = next_op_desc->Input(name)[0];
           if (var_name.compare(quant_out_var_name) == 0) {
              next_op_desc->SetInput(name,
                                     std::vector<std::string>({subgraph.at(int8_out)->Name()}));
          break;
        }
      }

       //remove the dequantize and quantize op
       GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out, quant_out});
       
       PADDLE_ENFORCE(subgraph.count(int8_out));
       IR_NODE_LINK_TO(subgraph.at(int8_out), next_op);

       found_squash_count++;
    }else{
       //Create an requantize Node
       OpDesc desc;
       desc.SetType("requantize");
       desc.SetInput("Input", std::vector<std::string>({subgraph.at(int8_out)->Name()}));
       desc.SetOutput("Output", std::vector<std::string>({quant_out->Name()}));
       desc.SetAttr("Scale_dequant", dequant_scale);
       desc.SetAttr("Scale_quant", quant_scale);
       desc.SetAttr("is_negative_input", is_negative);

       auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
       GraphSafeRemoveNodes(graph, {dequant, quant,  dequant_out});

       IR_NODE_LINK_TO(subgraph.at(int8_out), requant_node);
       IR_NODE_LINK_TO(requant_node, quant_out);

       found_squash_count++;
    }
};
  gpd(graph, handler);
  AddStatis(found_squash_count);
}

void CPUQuantizeSquashPass::DoubleBranch(Graph* graph) const{
  GraphPatternDetector gpd;
  auto* int8_out = gpd.mutable_pattern()
                ->NewNode("squash_double/int8_out")
                ->AsInput()
                ->assert_is_op_input("dequantize", "Input");

  patterns::DequantQuantRM squash_pattern(gpd.mutable_pattern(),"squash_double");
  int case_ = 2;
  squash_pattern(int8_out,case_);

  int found_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {

    VLOG(4) << "handle cpu quantize squash pass for dequantize-> two quantize";
    GET_IR_NODE_FROM_SUBGRAPH(dequant, dequantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant1, quantize1, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant2, quantize2, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op1, next_op1, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op2, next_op2, squash_pattern);


    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant1_out, quant1_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant2_out, quant2_out, squash_pattern);


    auto* dequant_op_desc = dequant->Op();
    auto* quant1_op_desc = quant1->Op();
    auto* quant2_op_desc = quant2->Op();
    float dequant_scale = boost::get<float>(dequant_op_desc->GetAttr("Scale"));
    float quant1_scale = boost::get<float>(quant1_op_desc->GetAttr("Scale"));
    float quant2_scale = boost::get<float>(quant2_op_desc->GetAttr("Scale"));
     
    if(dequant_scale == quant1_scale){

       if ( dequant_scale == quant2_scale){
           GraphSafeRemoveNodes(graph, {dequant, quant1, quant2, dequant_out, quant1_out, quant2_out});
 
           PADDLE_ENFORCE(subgraph.count(int8_out));
           IR_NODE_LINK_TO(subgraph.at(int8_out), next_op1);
           IR_NODE_LINK_TO(subgraph.at(int8_out), next_op2);
           found_squash_count++;
        }else{
           
       OpDesc desc;
       std::string squash_int8_out_in = subgraph.at(int8_out)->Name();
       std::string squash_out = quant2_out->Name();
       desc.SetInput("Input", std::vector<std::string>({squash_int8_out_in}));
       desc.SetOutput("Output", std::vector<std::string>({squash_out}));
       desc.SetAttr("Scale_dequant", dequant->Op()->GetAttr("Scale"));
       desc.SetAttr("Scale_quant", quant2->Op()->GetAttr("Scale"));
       desc.SetAttr("is_negative_input", quant2->Op()->GetAttr("is_negative_input"));
       desc.SetType("requantize");

       auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
       GraphSafeRemoveNodes(graph, {dequant, quant1,quant2,quant1_out,dequant_out});

       PADDLE_ENFORCE(subgraph.count(int8_out));
       IR_NODE_LINK_TO(subgraph.at(int8_out), requant_node);
       IR_NODE_LINK_TO(requant_node, quant2_out);
       IR_NODE_LINK_TO(subgraph.at(int8_out), next_op1);   
       found_squash_count++;
      }
    }else if(dequant_scale == quant2_scale){
    OpDesc desc;
       std::string squash_int8_out_in = subgraph.at(int8_out)->Name();
       std::string squash_out = quant1_out->Name();
       desc.SetInput("Input", std::vector<std::string>({squash_int8_out_in}));
       desc.SetOutput("Output", std::vector<std::string>({squash_out}));
       desc.SetAttr("Scale_dequant", dequant->Op()->GetAttr("Scale"));
       desc.SetAttr("Scale_quant", quant1->Op()->GetAttr("Scale"));
       desc.SetAttr("is_negative_input", quant1->Op()->GetAttr("is_negative_input"));
       desc.SetType("requantize");

       auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
       GraphSafeRemoveNodes(graph, {dequant, quant1,quant2,quant2_out,dequant_out});

       PADDLE_ENFORCE(subgraph.count(int8_out));
       IR_NODE_LINK_TO(subgraph.at(int8_out), requant_node);
       IR_NODE_LINK_TO(requant_node, quant1_out);
       IR_NODE_LINK_TO(subgraph.at(int8_out), next_op2);
       found_squash_count++;

   }else if (dequant_scale != quant2_scale){
      std::cout<<"Not supported"<<std::endl;
   }
};
  gpd(graph, handler);
  AddStatis(found_squash_count);

}

std::unique_ptr<ir::Graph> CPUQuantizeSquashPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("cpu_quantize_squash_pass", graph.get());
  
  SingleBranch(graph.get());
  DoubleBranch(graph.get());
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
  
  
