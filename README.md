# Make Densenet caffe protos and solver for imagenet or custom datasets
This contains the script to define densenet architectures including Densenet_121, Densenet_169, Densenet_201, Densenet_264 for imagenet and other custom datasets. This is truly flexible and efficient way to generate .prototxt files

## Setup and usage
### Dependencies
* Caffe [Installation Instructions](https://caffe.berkeleyvision.org/installation.html)
* Please remember to set the PYTHONPATH to $CAFFE_ROOT/python
### Usage
* Build lmdb from your dataset by changing the paths in the file convert_to_caffe_lmdb.py and running the following cmd:
```
python convert_to_caffe_lmdb.py
```
* Update the paths in the make_densenet.py file
* Run python make_densenet.py to generate the train_densenet.prototoxt and test_densenet.prototxt as well as solver.prototxt files
* Finally run the following command to start training on your own dataset
```
'/home/achyut/projects/caffe/build/tools/caffe' train \
  --solver='solver.prototxt' | tee training.log
```