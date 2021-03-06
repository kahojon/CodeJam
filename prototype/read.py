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
	def __init__(self, scale=False):
		self.REMISSED_PATIENTS = {}
		self.RESISTANT_PATIENTS = {}
		self.M_Rem_Pat = {}
		self.F_Rem_Pat = {}
		self.M_Res_Pat = {}
		self.F_Res_Pat = {}
		self.read()
		self.toFloat()
		if scale:
			self.scaleDown()
		

	def read(self):
		with open('trainingData.txt', "rU") as infile, open('trainingData.csv', 'wb') as outfile:
		    in_txt = csv.reader(infile, delimiter = '\t')
		    out_csv = csv.writer(outfile)
		    out_csv.writerows(in_txt)
		infile.close
		outfile.close

		with open('trainingData.csv', 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
			spamreader.next()
			spamreader.next()
			counter = 0
			for row in spamreader:
				for ind, var in enumerate(row[2:]):
					if var == 'COMPLETE_REMISSION':
						self.REMISSED_PATIENTS["Patient"+str(counter)]=row
					if var == 'RESISTANT':
						self.RESISTANT_PATIENTS["Patient"+str(counter)]=row
				counter = counter+1

		# # Parses male and female for remissed
		# for patient in self.REMISSED_PATIENTS:
		# 	for ind, var in enumerate(self.REMISSED_PATIENTS[patient]):
		# 		if var == 'M':
		# 			self.M_Rem_Pat[patient] = self.REMISSED_PATIENTS[patient]
		# 		if var == 'F':
		# 			self.F_Rem_Pat[patient] = self.REMISSED_PATIENTS[patient]

		# # Parses male and female for resistant
		# for patient in self.RESISTANT_PATIENTS:
		# 	for ind, var in enumerate(self.RESISTANT_PATIENTS[patient]):
		# 		if var == 'M':
		# 			self.M_Res_Pat[patient] = self.RESISTANT_PATIENTS[patient]
		# 		if var == 'F':
		# 			self.F_Res_Pat[patient] = self.RESISTANT_PATIENTS[patient]

		for patient in self.REMISSED_PATIENTS:
			for ind, var in enumerate(self.REMISSED_PATIENTS[patient]):
				# Yes/No responses
						if var in ('Yes','YES', 'COMPLETE_REMISSION', 'POS', 'F'):
							var = '1'
							self.REMISSED_PATIENTS[patient][ind] = var 
						if var in ('NotDone','No','NO', 'RESISTANT', 'NEG', 'M') :
							var='-1'
							self.REMISSED_PATIENTS[patient][ind] = var 
						# No information
						if var in ('NA', 'ND','NotDone'):
							var = '0'
							self.REMISSED_PATIENTS[patient][ind] = var  

						# Chemo type
						if var == 'Anthra-HDAC':
							var = '0'
							self.REMISSED_PATIENTS[patient][ind] = var  
						if var == 'HDAC-Plus':
							var = '1'
							self.REMISSED_PATIENTS[patient][ind] = var 
						if var == 'Flu-HDAC':
							var = '2'
							self.REMISSED_PATIENTS[patient][ind] = var 
						if var == 'StdAraC-Plus':
							var = '3'
							self.REMISSED_PATIENTS[patient][ind] = var 
						if var == 'Anthra-Plus':
							var ='4'
							self.REMISSED_PATIENTS[patient][ind] = var 

		for patient in self.RESISTANT_PATIENTS:
			for ind, var in enumerate(self.RESISTANT_PATIENTS[patient]):
				# Yes/No responses
						if var in ('YES', 'Yes', 'COMPLETE_REMISSION', 'POS', 'F'):
							var = '1'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						if var in ('No', 'NO','RESISTANT', 'NEG', 'M') :
							var='-1'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						# No information
						if var in ('NA', 'ND', 'NotDone'):
							var = '0'
							self.RESISTANT_PATIENTS[patient][ind] = var  

						# Chemo type
						if var == 'Anthra-HDAC':
							var = '0'
							self.RESISTANT_PATIENTS[patient][ind] = var  
						if var == 'HDAC-Plus':
							var = '1'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						if var == 'Flu-HDAC':
							var = '2'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						if var == 'StdAraC-Plus':
							var = '3'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						if var == 'Anthra-Plus':
							var ='4'
							self.RESISTANT_PATIENTS[patient][ind] = var 

	#Removes the patient ID				
	def toFloat(self):
		for x,y in self.REMISSED_PATIENTS.items():
			float_elements = []
			for m in y[1:]:
				float_elements.append(float(m))
				self.REMISSED_PATIENTS[x] = float_elements
		for x,y in self.RESISTANT_PATIENTS.items():
			float_elements = []
			for m in y[1:]:
				float_elements.append(float(m))
				self.RESISTANT_PATIENTS[x] = float_elements

	def get_trainset(self,split, label,vectorize = False):
		split = int(split*len(self.REMISSED_PATIENTS.items()))
		train_data = self.REMISSED_PATIENTS.items()[:split]
		test_data = self.REMISSED_PATIENTS.items()[split:]
		train_datares = self.RESISTANT_PATIENTS.items()[:split]
		test_datares = self.RESISTANT_PATIENTS.items()[split:]
		train_out =[y[label] for x,y in train_data]
		train_in = [y[:label] for x,y in train_data]
		train_temp = [x[label] for y,x in self.RESISTANT_PATIENTS.items()]
		train_tempin = [x[:label] for y,x in self.RESISTANT_PATIENTS.items()]
		train_out = train_out + train_temp
		if vectorize:
			for x in xrange(len(train_out)):
				if(train_out[x] == 1):
					train_out[x] = [1,0]
				elif(train_out[x] == -1):
					train_out[x] = [0,1]
		train_in = train_in + train_tempin
		trn_data = zip(train_out,train_in)
		# print trn_data
		tst_data = test_data + test_datares
		return trn_data,tst_data

	def scaleDown(self):
			print "Scaling"
			for x,y in self.REMISSED_PATIENTS.items():
				scaled = []
			ind = 1
			while (ind<267):
			 	total = 0
				for patient, y in self.REMISSED_PATIENTS.items():
					p = patient
					# print REMISSED_PATIENTS[patient][ind]
					total = total + (self.REMISSED_PATIENTS[patient][ind])
				
				average = total/len(self.REMISSED_PATIENTS)
				
				if average > 1:
					for patient in self.REMISSED_PATIENTS:
						# print REMISSED_PATIENTS[patient][ind]
						self.REMISSED_PATIENTS[patient][ind] = (self.REMISSED_PATIENTS[patient][ind] - average)/(len(self.REMISSED_PATIENTS)/2)

				ind = ind +1 

			i = 1
			while (i<267):
			 	total = 0
				for patient, y in self.RESISTANT_PATIENTS.items():
					p = patient
					# print REMISSED_PATIENTS[patient][ind]
					total = total + (self.RESISTANT_PATIENTS[patient][ind])
				
				average = total/len(self.RESISTANT_PATIENTS)
				
				if average > 1:
					for patient in self.RESISTANT_PATIENTS:
						# print REMISSED_PATIENTS[patient][ind]
						self.RESISTANT_PATIENTS[patient][ind] = (self.RESISTANT_PATIENTS[patient][ind] - average)/(len(self.REMISSED_PATIENTS)/2)
				i = i+1


			

