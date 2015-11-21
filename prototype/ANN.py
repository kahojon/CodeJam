import random
import math
import numpy as np
class NeuralNet(object):
        def __init__(self,layers):

                self.layers = len(layers)
                self.neurons_in_layer = layers
                self.bias =[np.random.randn(x,1) for x in layers[1:]]
                self.weights =[np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])]

        def feed(self,x):

                act = x
                acts = [x]        
                zv=[]
                for b,w in zip(self.bias,self.weights):
                    z = np.dot(w,act)+b
                    act = self.sigmoid(z)
                    acts.append(act)
                return acts[-1]

        def train(self, minibatch_s,train_set,learn_rate,epoch):

                for e in xrange(epoch):
                    random.shuffle(train_set)
                    minibatches = [train_set[k:k+minibatch_s]for k in xrange(0,len(train_set),minibatch_s)]
                    for mini_batch in minibatches:
                        self.update(learn_rate,minibatch_s,mini_batch)
                       # s= str(eval(minibatch))
                       # print s
                    print e
               # for w,b in zip(self.weights,self.bias):
                #    print "bias",b
                 #   print "weights",w

        def update(self,learn_rate,minibatch_size,mini_batch):

                nb = [np.zeros(b.shape)for b in self.bias]
                nw = [np.zeros(w.shape)for w in self.weights]
                #print nw
                for y,x in mini_batch:
                    nweights,nbias = self.back_prop(x,y)
                    #print nweights
                    #print zip(nweights,nw)
                    #print "-------------------------------------"
                    #print self.weights
                    nw = [a+aa for a, aa in zip(nweights,nw)]
                    nb = [b+bb for b, bb in zip(nbias,nb)]
#                    print self.weights
                    #print nw
                    
                rate = learn_rate/minibatch_size
                self.weights = [sw -rate*n for sw, n in zip(self.weights,nw)]
                self.bias = [sb -rate*b for sb ,b in zip(self.bias,nb) ]
           
        def back_prop(self,inp,output):
                #print inp
                print output.shape
                nweights = [np.zeros(w.shape) for w in self.weights]
                nbias = [np.zeros(b.shape) for b in self.bias]
                activation = np.array(inp)
                activations = [activation]
                zv = []
                for x,y in zip(self.bias,self.weights):
                        z = np.dot(y,activation)+x
                        zv.append(z)
                        activation = self.sigmoid(z)
                        activations.append(activation)
#                print zv
#                print activations
                output = np.array(output)
                cd = self.cost_derivative(activations[-1],output)
                delta = np.multiply(cd,self.sigmoidpr(zv[-1]))
#                print activations[-2].transpose()
#                print delta
                #print activations[-2].shape
                #print delta.shape
                nweights[-1] =  np.dot(delta,activations[-2].transpose())
                nbias [-1] = delta
                ds = [delta]
                for l in range(2,self.layers):
                   m = np.dot(self.weights[-l+1].transpose(),delta)
                   n = self.sigmoidpr(zv[-l])
                   delta = np.multiply(m,n)
                   ds.append(delta)
                   #print delta
                   #print activations[-l-1]
                   nweights[-l] = np.dot(delta,activations[-l-1].transpose())
                   nbias[-l] = delta
#                print nweights    
                return (nweights,nbias)
        def eval(self,inp):
           perc = []
           for i,x in inp:
             res = self.feed(inp)
             perc.append((x - out)/x)
           err = sum(perc)/len(perc)  
           return err 
        def cost_derivative(self,a,y):
                return (a-y)
        def sigmoid(self,x):
                return 1/(1+np.exp(-x))
        def sigmoidpr(self,x):
                return (1-self.sigmoid(x))*(self.sigmoid(x))
from read import getData as gd
if __name__ =='__main__':
    
    g = gd()
    train_data = g.get_trainset(0.8,265,vectorize = True)
    train_set = train_data[0]
    test_set = train_data[1]
    n = NeuralNet([265,100,100,2])
    n.train(10,train_set,0.01,100)
    print n.feed(test_set[0][1])
    print n.feed(test_set[0][0])


              
