include(operators)
register_operators(DEPS naive_executor)
cc_library(op_variant SRCS op_variant.cc DEPS operator proto_desc)
cc_library(conditional_block_op_helper SRCS conditional_block_op_helper.cc DEPS operator op_variant conditional_block_op)
cc_library(recurrent_op_helper SRCS recurrent_op_helper.cc DEPS operator op_variant recurrent_op)
cc_library(while_op_helper SRCS while_op_helper.cc DEPS operator op_variant) 

target_link_libraries(conditional_block_infer_op conditional_block_op) 

file(APPEND ${pybind_file} "USE_OP(less_than);\nUSE_OP(logical_and);\nUSE_NO_KERNEL_OP(read_from_array);\n")
