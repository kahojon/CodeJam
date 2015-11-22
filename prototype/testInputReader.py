#!/usr/bin/env python
import sys, argparse, csv
import numpy as np


#269 columns
# No = -1, Yes = 1, Complete_remission = 1, resilience = 0, Everything else = -1
# NA=0
#NEG=-1, POS=1
# F = 1, M=-1
# Chemos: ANTHRA_HDAC = 0, HDAC-PLUS=1, FLU_HDAC = 2; STDARAC-PLUS = 3
class getData:
	def __init__(self):
		self.patients = {}
		self.ids=[]

	def read(self, filename='stdin'):
		name = filename
		if filename.endswith(".txt"):
			name = filename[0:-4]
		if filename.endswith(".in"):
			name = filename[0:-3]
		with open(filename, "rU") as infile, open(name + '.csv', 'wb') as outfile:
		    in_txt = csv.reader(infile, delimiter = '\t')
		    out_csv = csv.writer(outfile)
		    out_csv.writerows(in_txt)
		infile.close
		outfile.close

		# Store patients into a dictionary based on name
		with open(name+'.csv', 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			spamreader.next()
			spamreader.next()
			for row in spamreader:
				for ind, var in enumerate(row[2:]):
					self.patients[row[0]]=row[1:]
	
		# Converts strings to matching string numbers
		for patient, variables in self.patients.items():
			for ind, var in enumerate(variables):
				if var in ('Yes','YES', 'COMPLETE_REMISSION', 'POS', 'F'):
					var = '1'
					self.patients[patient][ind] = var 
				if var in ('NotDone','No','NO', 'RESISTANT', 'NEG', 'M') :
					var='-1'
					self.patients[patient][ind] = var 
				# No information
				if var in ('NA', 'ND','NotDone'):
					var = '0'
					self.patients[patient][ind] = var  

				# Chemo type
				if var == 'Anthra-HDAC':
					var = '0'
					self.patients[patient][ind] = var  
				if var == 'HDAC-Plus':
					var = '1'
					self.patients[patient][ind] = var 
				if var == 'Flu-HDAC':
					var = '2'
					self.patients[patient][ind] = var 
				if var == 'StdAraC-Plus':
					var = '3'
					self.patients[patient][ind] = var 
				if var == 'Anthra-Plus':
					var = '4'
					self.patients[patient][ind] = var
	
	# Converts string numbers to float
	def to_float(self):
		for x,y in self.patients.items():
			float_elements = []
			for m in y[0:]:
				float_elements.append(float(m))
				self.patients[x] = float_elements
		
		# SAVE TO ARRAY TO BE PRINTED		
		for x,y in self.patients:
			(self.ids).append(x)

	# Scaling test set
	def scale_down(self):
		length = -1
		for x, y in self.patients.items():
			length = len(y)
		ind = 1
		while (ind < length):
		 	total = 0
			for patient, y in self.patients.items():
				total = total + self.patients[patient][ind]
			
			average = total/len(self.patients)
			
			if average > 1:
				for patient in self.spatients:
					# print REMISSED_PATIENTS[patient][ind]
					self.patients[patient][ind] = (self.patients[patient][ind] - average)/(len(self.patients)/2)
		ind = ind +1 
	
	def runtime(self,label,vectorize = False):
		model_data = self.patients.items()
		model_set = [x[:label] for y,x in model_data]
		id_data = zip(self.ids,model_set)
		return id_data

# Get test set
# def get_testset(self,split, label,vectorize = False):
# 	split = int(split*len(patients.items()))
# 	train_data = patients.items()[:split]
# 	test_data = patients.items()[split:]
# 	train_datares = self.RESISTANT_PATIENTS.items()[:split]
# 	test_datares = self.RESISTANT_PATIENTS.items()[split:]
# 	train_out =[y[label] for x,y in train_data]
# 	train_in = [y[:label] for x,y in train_data]
# 	train_temp = [x[label] for y,x in self.RESISTANT_PATIENTS.items()]
# 	train_tempin = [x[:label] for y,x in self.RESISTANT_PATIENTS.items()]
# 	train_out = train_out + train_temp
# 	if vectorize:
# 		for x in xrange(len(train_out)):
# 			if(train_out[x] == 1):
# 				train_out[x] = [1,0]
# 			elif(train_out[x] == -1):
# 				train_out[x] = [0,1]
# 	train_in = train_in + train_tempin
# 	trn_data = zip(np.asarray(train_out),np.asarray(train_in))
# 	tst_data = test_data + test_datares
# 	tst_in = [x[:label] for y,x in tst_data]
# 	tst_out = [x[label] for y,x in tst_data]
# 	tst = zip(np.asarray(tst_out),np.asarray(tst_in))
# 	return trn_data, tst, self.ids



