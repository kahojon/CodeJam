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
		self.REMISSED_PATIENTS = {}
		self.RESISTANT_PATIENTS = {}
		self.M_Rem_Pat = {}
		self.F_Rem_Pat = {}
		self.M_Res_Pat = {}
		self.F_Res_Pat = {}
		self.read()
		self.toFloat()

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

		# Parses male and female for remissed
		for patient in REMISSED_PATIENTS:
			for ind, var in enumerate(REMISSED_PATIENTS[patient]):
				if var == 'M':
					M_Rem_Pat[patient] = REMISSED_PATIENTS[patient]
				if var == 'F':
					F_Rem_Pat[patient] = REMISSED_PATIENTS[patient]

		# Parses male and female for resistant
		for patient in RESISTANT_PATIENTS:
			for ind, var in enumerate(RESISTANT_PATIENTS[patient]):
				if var == 'M':
					M_Res_Pat[patient] = RESISTANT_PATIENTS[patient]
				if var == 'F':
					F_Res_Pat[patient] = RESISTANT_PATIENTS[patient]

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

		for patient in self.RESISTANT_PATIENTS:
			for ind, var in enumerate(self.RESISTANT_PATIENTS[patient]):
				# Yes/No responses
						if var in ('YES', 'COMPLETE_REMISSION', 'POS', 'F'):
							var = '1'
							self.RESISTANT_PATIENTS[patient][ind] = var 
						if var in ('No', 'RESISTANT', 'NEG', 'M') :
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
	#Removes the patient ID				
	def toFloat(self):
		for x,y in self.REMISSED_PATIENTS.items():
			float_elements = []
			for m in y[1:]:
				float_elements.append(float(m))
				self.REMISSED_PATIENTS[x] = float_elements

	def scaleDown(self):
<<<<<<< HEAD
		for x,y in self.REMISSED_PATIENTS.items():
			scaled = []
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
		print trn_data
		tst_data = test_data + test_datares
		return trn_data,tst_data
=======
		ind = 1
		while (ind<len(self.REMISSED_PATIENTS["Patient166"])):
		 	total = 0
			for patient, y in self.REMISSED_PATIENTS.items():
				p = patient
				# print REMISSED_PATIENTS[patient][ind]
				total = total + (REMISSED_PATIENTS[patient][ind])
			
			average = total/len(self.REMISSED_PATIENTS)
			
			if average > 1:
				for patient in self.REMISSED_PATIENTS:
					# print REMISSED_PATIENTS[patient][ind]
					REMISSED_PATIENTS[patient][ind] = (REMISSED_PATIENTS[patient][ind] - average)/(len(REMISSED_PATIENTS)/2)

			ind = ind +1 

		i = 1
		while (ind<len(self.RESISTANT_PATIENTS["Patient165"])):
		 	total = 0
			for patient, y in self.RESISTANT_PATIENTS.items():
				p = patient
				# print REMISSED_PATIENTS[patient][ind]
				total = total + (RESISTANT_PATIENTS[patient][ind])
			
			average = total/len(self.RESISTANT_PATIENTS)
			
			if average > 1:
				for patient in self.RESISTANT_PATIENTS:
					# print REMISSED_PATIENTS[patient][ind]
					REMISSED_PATIENTS[patient][ind] = (REMISSED_PATIENTS[patient][ind] - average)/(len(REMISSED_PATIENTS)/2)
			i = i+1

	def get_trainset(self,split):
		split = int(split*len(REMISSED_PATIENTS.items()))
		train_data = REMISSED_PATIENTS.items()[:split]
		test_data = REMISSED_PATIENTS.items()[split:]
		train_out =[y[266] for x,y in train_data]
		train_in = [x[:266] for x,y in train_data]
		trn_data = zip(train_out,train_in)
		return trn_data
>>>>>>> 070ded8b6df949f7b4d23416e952fdbe04ce50ac

if __name__ == "__getData__":
	read()
	toFloat()
	scaleDown()
			

