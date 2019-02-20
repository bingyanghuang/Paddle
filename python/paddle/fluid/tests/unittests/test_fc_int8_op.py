# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest


def fc_refer(matrix):
    in_n, in_h, in_w = matrix.input.shape
    w_i, w_o = matrix.weights.shape

    x_data = np.reshape(matrix.input, [in_n, in_h * in_w])
    w_data = np.reshape(matrix.weights, [w_i, w_o])
    result = None

    result = np.dot(x_data, w_data)

    return result


class MatrixGenerate:
    def __init__(self, mb, oc, h, w):
        self.input = np.random.random((mb,  h, w)).astype("float32")
        self.weights = np.random.random(( h * w, oc)).astype("float32")
        self.bias = np.random.random((1, oc)).astype("float32")

class TestFCOp(OpTest):
    def setUp(self):
        self.op_type = "fc"
        self.matrix = MatrixGenerate(1, 1, 1, 2)

        self.inputs = {'Input': self.matrix.input, 'W': self.matrix.weights}

        print(self.inputs)
        self.attrs = {'use_mkldnn': False}

        self.outputs = {'Out': fc_refer(self.matrix)}

    def test_check_output(self):
        self.check_output()

if __name__ == "__main__":
    unittest.main()
                                                                      
