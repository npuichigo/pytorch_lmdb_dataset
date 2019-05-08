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
from __future__ import unicode_literals

import argparse
import lmdb
import numpy as np

from proto import utils
from proto import tensor_pb2


def create_db(output_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    print(LMDB_MAP_SIZE)
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    checksum = 0
    with env.begin(write=True) as txn:
        for j in range(0, 1024):
            # MODIFY: add your own data reader / creator
            width = 64
            height = 32
            img_data = np.random.rand(3, width, height).astype(np.float32)
            label = np.asarray(j % 10)

            # Create TensorProtos
            tensor_protos = tensor_pb2.TensorProtos()
            img_tensor = utils.numpy_array_to_tensor(img_data)
            tensor_protos.protos.extend([img_tensor])

            label_tensor = utils.numpy_array_to_tensor(label)
            tensor_protos.protos.extend([label_tensor])
            txn.put(
                '{}'.format(j).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            if (j % 16 == 0):
                print("Inserted {} rows".format(j))


def main():
    parser = argparse.ArgumentParser(
        description="LMDB creation"
    )
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to write the database to",
                        required=True)
    args = parser.parse_args()

    create_db(args.output_file)


if __name__ == '__main__':
    main()
