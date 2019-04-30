# Copyright 2019 ASLP@NPU.  All rights reserved.
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
#
# Author: npuichigo@gmail.com (zhangyuchao)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset

from proto import tensor_pb2


def _parse_tensor_proto(tensor_proto):
    if tensor_proto.data_type == tensor_pb2.TensorProto.FLOAT:
        tensor = np.array(tensor_proto.float_data)
    elif tensor_proto.data_type == tensor_pb2.TensorProto.INT32:
        tensor = np.array(tensor_proto.int32_data)
    else:
        raise ValueError("Only float and int32 data are supported now")
    return tensor.reshape(tensor_proto.dims)


class LmdbDataset(Dataset):
    """Lmdb dataset."""

    def __init__(self, lmdb_path):
        super(LmdbDataset, self).__init__()
        import lmdb
        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key, _ in txn.cursor()]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            serialized_str = txn.get(self.keys[index])
        tensor_protos = tensor_pb2.TensorProtos()
        tensor_protos.ParseFromString(serialized_str)
        img = _parse_tensor_proto(tensor_protos.protos[0])
        label = _parse_tensor_proto(tensor_protos.protos[1])
        return img, label

    def __len__(self):
        return self.length
