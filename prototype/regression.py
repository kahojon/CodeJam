import numpy as np
import random
import math
class RegModel(object):
	def __init__(self,feature_size):
		self.parameters = np.random.randn(feature_size+1,1)
		self.old_parameters = np.zeros(self.parameters.shape)
		self.cost = 100

	def train(self, train_set, eta,epoch):
		random.shuffle(train_set)
		for e in xrange(epoch):
			for y,x in train_set:
				x = np.insert(x,0,1)
				x = np.array(x)
				self.update(eta,y,x)
				#print self.cost_func(train_set)

			print e
		
			
	def update(self,eta,output,input):
		newp = np.zeros(self.parameters.shape)
		for j in xrange(len(input)):
			newp = eta*(self.cost_derivative(input,output)*input[j])
			self.old_parameters[j] = self.parameters[j]
			self.parameters[j] = self.parameters[j] + newp

	def cost_derivative(self,x,y):
		return (y - self.reg(x))
	def cost_func(self,train_set):
		suma = []
		for y,x in train_set:
			x = np.insert(x,0,1)
			a = 1.0/2.0 *(y-self.reg(x))**2
			suma.append(a)
		sum = np.sum(suma)
		return 1.0/len(train_set) * sum
	def reg(self,x):
		output = np.dot(self.parameters.transpose(),x)
		#print output
		return output

from read import getData as gd
if __name__ =='__main__':
	g = gd(scale=True)
	train_data = g.get_trainset(0.8,266)
	train_set = train_data[0]
	test_set = train_data[1]
	n = RegModel(266)
	y = test_set[0][1]
	y = np.insert(y,0,1)
	n.train(train_set,0.000001,1000)
	x = test_set[0][0]
	print n.reg(y)
	print x
	with open("parameters.txt",'w') as f:
		f.write(np.array_str(n.parameters))