import os
import sys
import caffe
from pylab import *

from caffe import layers as L
from caffe import params as P

TRAINING_FILENAME = 'data/train_h5_list.txt'#data/train.txt'
TESTING_FILENAME = 'data/train_h5_list.txt'#'data/val.txt'

def lenet(image_filename, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(batch_size=batch_size, source=image_filename)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=125, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()
    
#with open('lenet_auto_train.prototxt', 'w') as f:
#    print lenet(TRAINING_FILENAME, 64)
#    f.write(str(lenet(TRAINING_FILENAME, 64)))
    
#with open('lenet_auto_test.prototxt', 'w') as f:
#    f.write(str(lenet(TESTING_FILENAME, 100)))

# SETTING CAFFE
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('lenet_solver.prototxt')
# TESTING FORWARD
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

niter = 10000
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            #print 'Prediction'
            #print solver.test_nets[0].blobs['ip2'].data.argmax(1)
            #print 'label'
            #print solver.test_nets[0].blobs['label'].data
            correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                           == solver.test_nets[0].blobs['y0'].data)
        test_acc[it // test_interval] = correct /float(2400) 
        print 'Prediction'
        print solver.test_nets[0].blobs['ip2'].data.argmax(1)
        print 'label'
        print solver.test_nets[0].blobs['label'].data
        print "Loss of current test batch:"+str(solver.test_nets[0].blobs['loss'].data)
        print correct /float(2400)
        print test_acc
