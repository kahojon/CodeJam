import random
import argparse
import csv
import sys

import numpy as np


#269 columns
# No = -1, Yes = 1, Complete_remission = 1, resilience = 0, Everything else = -1
# NA=0
#NEG=-1, POS=1
# F = 1, M=-1
# Chemos: ANTHRA_HDAC = 0, HDAC-PLUS=1, FLU_HDAC = 2; STDARAC-PLUS = 3
class getTestData:
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


#269 columns
# No = -1, Yes = 1, Complete_remission = 1, resilience = 0, Everything else = -1
# NA=0
#NEG=-1, POS=1
# F = 1, M=-1
# Chemos: ANTHRA_HDAC = 0, HDAC-PLUS=1, FLU_HDAC = 2; STDARAC-PLUS = 3
class getTrainData():
    def __init__(self):
        self.REMISSED_PATIENTS = {}
        self.RESISTANT_PATIENTS = {}
        self.M_Rem_Pat = {}
        self.F_Rem_Pat = {}
        self.M_Res_Pat = {}
        self.F_Res_Pat = {}
        self.read()
        self.toFloat()

    def read(self, filename='trainingData.txt'):
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
        #   for ind, var in enumerate(self.REMISSED_PATIENTS[patient]):
        #       if var == 'M':
        #           self.M_Rem_Pat[patient] = self.REMISSED_PATIENTS[patient]
        #       if var == 'F':
        #           self.F_Rem_Pat[patient] = self.REMISSED_PATIENTS[patient]

        # # Parses male and female for resistant
        # for patient in self.RESISTANT_PATIENTS:
        #   for ind, var in enumerate(self.RESISTANT_PATIENTS[patient]):
        #       if var == 'M':
        #           self.M_Res_Pat[patient] = self.RESISTANT_PATIENTS[patient]
        #       if var == 'F':
        #           self.F_Res_Pat[patient] = self.RESISTANT_PATIENTS[patient]

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
        trn_data = zip(np.asarray(train_out),np.asarray(train_in))
        # print trn_data
        tst_data = test_data + test_datares
        tst_in = [x[:label] for y,x in tst_data]
        tst_out = [x[label] for y,x in tst_data]
        tst = zip(np.asarray(tst_out),np.asarray(tst_in))
        return trn_data,tst

    def scaleDown(self):
        for x,y in self.REMISSED_PATIENTS.items():
            scaled = []
        ind = 1
        while (ind<len(self.REMISSED_PATIENTS["Patient166"])):
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
        while (ind<len(self.RESISTANT_PATIENTS["Patient165"])):
            total = 0
            for patient, y in self.RESISTANT_PATIENTS.items():
                p = patient
                # print REMISSED_PATIENTS[patient][ind]
                total = total + (self.RESISTANT_PATIENTS[patient][ind])
            
            average = total/len(self.RESISTANT_PATIENTS)
            
            if average > 1:
                for patient in self.RESISTANT_PATIENTS:
                    # print REMISSED_PATIENTS[patient][ind]
                    self.REMISSED_PATIENTS[patient][ind] = (self.REMISSED_PATIENTS[patient][ind] - average)/(len(self.REMISSED_PATIENTS)/2)
            i = i+1


####### NEURAL NET #########

class CrossEntropyCost(object):

    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def cost_der(z, a, y):
        return (a-y[:,None])

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed(self, a):
        a = a[None,:].transpose()
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, training_data, epochs, mini_batch_s, eta):
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_s]
                for k in xrange(0, n, mini_batch_s)]
            for mini_batch in mini_batches:
                self.update(
                    mini_batch, eta, len(training_data))
#            print "Epoch %s of %s training complete" % (j, str(epochs))

    def update(self, mini_batch, eta, n):
        nbs = [np.zeros(b.shape) for b in self.biases]
        nws = [np.zeros(w.shape) for w in self.weights]
        for y, x in mini_batch:
            delta_nbs, delta_nws = self.backprop(x, y)
            nbs = [nb+dnb for nb, dnb in zip(nbs, delta_nbs)]
            nws = [nw+dnw for nw, dnw in zip(nws, delta_nws)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nws)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nbs)]

    def backprop(self, x, y):
        x = x[None,:].transpose()
        nbs = [np.zeros(b.shape) for b in self.biases]
        nws = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).cost_der(zs[-1], activations[-1], y)
        nbs[-1] = delta
        nws[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nbs[-l] = delta
            nws[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nbs, nws)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


############# MAIN ###############

def getIds():
    import getTestData as test_gd
    g2 = test_gd()
    id_data = g2.runtime(265,vectorize=True)
    patient_id = id_data[0]
    #print id_data
    return patient_id 

def problem1():
    #input training data csv
    import getTrainData as gd
    g = gd()

    train_set = g.get_trainset(1.0,265,vectorize=True)
 #   train_set = g.get_trainset(0.8,265,vectorize=True)
    train_data = train_set[0]
 #   test_data = train_set[1]

    import getTestData as test_gd
    g2 = test_gd()
    test_set = g2.runtime(265,vectorize=True)
    test_data = test_set[1]

    import Network as nw
    n = nw([265,100,100,2])
    n.train(train_data,100,20,0.0001)

    remission_status = []

    for i in xrange(len(test_data)):
            result = n.feed(test_data[i][1])
            result_list = result.tolist()
    
    for j in range(0,len(result_list)):        
            if (result_list[j] > 0.5) :
                remission_status.append("COMPLETE_REMISSION")
            elif (result_list[j+1] > 0.5) :
                remission_status.append("RESISTANT")
            else :
                print "ERROR PROB 1"
            j+=1

    return remission_status

def problem2():
    remission_duration = []
    return remission_duration

def problem3():
    overall_survival = []
    return overall_survival 

def main():
    patient_id = getIds()
    remission_status = problem1()
    remission_duration = problem2()
    overall_survival = problem3()

    for i in range(0, len(patient_id)):
        line = patient_id[i]
        try: 
            line += " " + remission_status[i]
        except IndexError:
            line += " 0 "
        try:
            line += " " + remission_duration[i]
        except IndexError:
            line += " 0 "
        try:
            line += " " + overall_survival[i]
        except IndexError:
            line += " 0 "
        print line 
        sys.stdout.write(line)

if __name__ == '__main__':
    main()

