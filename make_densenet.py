from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride, 
                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    if dropout>0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv

def dense_block(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    conv = bn_relu_conv(bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv, axis=1)
    return concate

def transition_block(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return pooling

# The following is based on the official Densenet paper architecture for Imagenet
#dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def densenet(data_file, architecture = 'densenet_121', mode='train', batch_size=64, no_of_classes = 3, dropout=0.2):
    
    architecture_map = dict({'densenet_121':[6, 12, 24, 16], 'densenet_169':[6, 12, 32, 32], 'densenet_201':[6, 12, 48, 32], 'densenet_264':[6, 12, 64, 48]})

    no_dense_blocks_ar = architecture_map[architecture]

    data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
              transform_param=dict(mean_file="/home/achyut/Desktop/classifier_sorted_rgb/lmdb/mean.binaryproto"))

    nchannels = 112
    model = L.Convolution(data, kernel_size=7, stride=2, num_output=nchannels,
                        pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    model = L.Pooling(model, pool=P.Pooling.MAX, kernel_size=3, stride=2)


    nchannels = first_output/2

    for i in range(no_of_dense_blocks_ar[0]):
        model = dense_block(model, nchannels, dropout)
    model = transition_block(model, nchannels, dropout)
    nchannels /= 2

    for i in range(no_of_dense_blocks_ar[1]):
        model = dense_block(model, nchannels, dropout)
    model = transition_block(model, nchannels, dropout)
    nchannels /= 2

    for i in range(no_of_dense_blocks_[2]):
        model = dense_block(model, nchannels, dropout)
    model = transition_block(model, nchannels, dropout)
    nchannels /= 2

    for i in range(no_of_dense_blocks_ar[3]):
        model = dense_block(model, nchannels, dropout)
    model = transition_block(model, nchannels, dropout)
    nchannels /= 2


    model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    model = L.InnerProduct(model, num_output=no_of_classes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)
    return to_proto(loss, accuracy)

def make_net(train_lmdb_path = '/home/achyut/Desktop/classifier_sorted_rgb/lmdb/train_lmdb', test_lmdb_path = '/home/achyut/Desktop/classifier_sorted_rgb/lmdb/test_lmdb', train_batch_size = 64, test_batch_size = 32):

    with open('train_densenet.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
        print(str(densenet(train_lmdb_path, batch_size=train_batch_size)), file=f)

    with open('test_densenet.prototxt', 'w') as f:
        print(str(densenet(test_lmdb_path, batch_size=test_batch_size)), file=f)

def make_solver(max_iter = 230000, base_lr = 0.05):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_densenet.prototxt'
    s.test_net.append('test_densenet.prototxt')
    s.test_interval = 800
    s.test_iter.append(200)

    s.max_iter = max_iter
    s.type = 'Adam'
    s.display = 1

    s.base_lr = base_lr
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.25 * s.max_iter))
    s.stepvalue.append(int(0.50 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net(train_lmdb_path = '/home/achyut/Desktop/classifier_sorted_rgb/lmdb/train_lmdb', test_lmdb_path = '/home/achyut/Desktop/classifier_sorted_rgb/lmdb/test_lmdb', train_batch_size = 64, test_batch_size = 32)
    make_solver(max_iter = 230000, base_lr = 0.05)










