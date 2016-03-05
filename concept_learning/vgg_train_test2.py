import os
import sys
import caffe
from pylab import *
import time

from caffe import layers as L
from caffe import params as P

#TRAINING_FILENAME = 'data/train_h5_list.txt'#data/train.txt'
#TESTING_FILENAME = 'data/train_h5_list.txt'#'data/val.txt'

#with open('lenet_auto_train.prototxt', 'w') as f:
#    print lenet(TRAINING_FILENAME, 64)
#    f.write(str(lenet(TRAINING_FILENAME, 64)))
    
#with open('lenet_auto_test.prototxt', 'w') as f:
#    f.write(str(lenet(TESTING_FILENAME, 100)))
loss_interval = 10
t = time.strftime("%H:%M:%S")
log_file = open('log_'+t+'.txt','w')


# SETTING CAFFE
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('vggnet_solver_adagrad.prototxt')
solver.net.copy_from('VGG_CNN_S.caffemodel')
# TESTING FORWARD
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

niter = 10000
test_interval = 50
# losses will also be stored in the log
#train_loss = zeros(niter)
#test_acc = zeros(int(np.ceil(niter / test_interval)))
label_list = ["landscape","wildlife","travel","vacation","sunrise","sunset","night","art","architecture","urban","abandoned","beautiful","cute","love","beauty","summer","autumn","winter","spring"]
# the main solver loop
loss = [[] for i in xrange(19)]
accuracies = [[] for i in xrange(19)]
balanced = [[] for i in xrange(19)]
TPRs = [[] for i in xrange(19)]
precs = [[] for i in xrange(19)]
testrange = 1000
try:
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
    
    
    	# store the train loss
   	#train_loss[it] = solver.net.blobs['loss'].data
    	if it % loss_interval == 0:
	    for idx,concept in enumerate(label_list):
	    	#loss[idx].append(solver.net.blobs['loss_'+concept].data)
		c_loss = str(solver.net.blobs['loss_'+concept].data)
		print c_loss
		loss[idx].append(c_loss)
	# store the output on the first test batch
    	# (start the forward pass at conv1 to avoid loading new data)
    	solver.test_nets[0].forward(start='conv1')
    
    	no_concepts = 19
    	# run a full test every so often
    	# (Caffe can also do this for us and write to a log, but we show here
    	#  how to do it directly in Python, where more complicated things are easier.)
    	if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = np.zeros((no_concepts))
	    true_positives = np.zeros((no_concepts))
	    pred_positives = np.zeros((no_concepts))
	    true_negatives = np.zeros((no_concepts))
	    pred_negatives = np.zeros((no_concepts))
	    TPR = np.zeros((no_concepts))
            TPR0 = np.zeros((no_concepts))
            for test_it in range(testrange):
                solver.test_nets[0].forward()

	        for idx,concept in enumerate(label_list):
		
		    preds = solver.test_nets[0].blobs['fc9_'+concept].data.argmax(1)
		    inv_preds = np.ones((24))-preds
                    trues = solver.test_nets[0].blobs['label_'+concept].data
		    inv_trues = np.ones((24))-trues
	    	    TPR[idx] += preds.dot(trues)
		    TPR0[idx] += inv_preds.dot(inv_trues)
		    true_positives[idx]+= sum(trues)
		    pred_positives[idx]+= sum(preds)
		    true_negatives[idx]+= sum(inv_trues)
		    pred_negatives[idx]+= sum(inv_preds)
		    correct[idx]+=sum(solver.test_nets[0].blobs['fc9_'+concept].data.argmax(1)==solver			.test_nets[0].blobs['label_'+concept].data)
            print "Metrics..."
	    print "Pos. Predictions",pred_positives
	    print "True Positives",true_positives
            print "Neg Predictions",pred_negatives
	    print "True Negatives",true_negatives
	    sens = [TPR[idx]/float(true_positives[idx]) for idx in xrange(19)]
		
	    print "TPR",sens
	    sens0 = [TPR0[idx]/float(true_negatives[idx]) for idx in xrange(19)]
            print 'TPR0',sens0
	    print "Accuracy",correct/float(24*testrange)
	    prec = [TPR[idx]/float(pred_positives[idx]) for idx in xrange(19)]
	    balance = [(sens[idx]+TPR0[idx]/float(true_negatives[idx]))/float(2.0) for idx in xrange(19)]
	    print "Balanced Accuracy",balance
            print "Precision",prec
	    for idx in xrange(19):
	        accuracies[idx].append(correct[idx]/float(24*testrange))
		balanced[idx].append(balance[idx])
	        TPRs[idx].append(sens[idx])
	        precs[idx].append(prec[idx])

except KeyboardInterrupt:
    for idx,concept in enumerate(label_list):
        concept_loss = ','.join([str(elem) for elem in loss[idx]])
	log_file.write(concept+'=['+concept_loss+']\n')
    print "Accuracies"
    for idx,concept in enumerate(label_list):   
        log_file.write(concept+'=['+','.join([str(elem) for elem in accuracies[idx]])+']\n')
    print "TPR"
    for idx,concept in enumerate(label_list):
        log_file.write(concept+'=['+','.join([str(elem) for elem in TPRs[idx]])+']\n')
    print 'Precisions'
    for idx,concept in enumerate(label_list):
        log_file.write(concept+'=['+','.join([str(elem) for elem in precs[idx]])+']\n')
    log_file.write('Balanced\n')
    for idx,concept in enumerate(label_list):
	log_file.write(concept+'=['+','.join([str(elem) for elem in balanced[idx]])+']\n')    
    #log_file.write(TPRs)
    #log_file.write(precs)
    log_file.close()
except:
    print "error"
