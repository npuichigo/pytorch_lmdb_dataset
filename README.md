# Pytorch Lmdb Dataloader
Use lmdb with protobuf to efficiently read big data for pytorch training.

## Getting Started
1. Install python and protobuf. It's convinient to get protoc in grpc_tools.
```shell
pip install grpcio grpcio-tools
```
2. Generate proto.
```shell
python -m grpc_tools.protoc -I./proto --python_out=./proto ./proto/tensor.proto
```
3. Create dummy training data.
```shell
python create_lmdb.py --output_file train_lmdb
```
4. Run the unit testing.
```shell
python dataset_test.py
```

## Reference
* https://discuss.pytorch.org/t/whats-the-best-way-to-load-large-data/2977
* https://github.com/pytorch/pytorch/blob/master/caffe2/python/examples/lmdb_create_example.py
