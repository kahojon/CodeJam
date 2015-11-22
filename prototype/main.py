import random
import sys

import numpy as np

def getIds():
    from testInputReader import getData as test_gd
    g2 = test_gd()
    id_data = g2.runtime(265,vectorize=True)
    patient_id = id_data[0]
    return patient_id 

def problem1():
    #input training data csv
    from trainingReader import getData as gd
    g = gd()

    train_set = g.get_trainset(1.0,265,vectorize=True)
 #   train_set = g.get_trainset(0.8,265,vectorize=True)
    train_data = train_set[0]
 #   test_data = train_set[1]

    from testInputReader import getData as test_gd
    g2 = test_gd()
    test_set = g2.runtime(265,vectorize=True)
    test_data = test_set[1]

    from network2 import Network as nw
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
        if remission_status[i] is not None:
            line += " " + remission_status[i]
    #    if remission_duration[i] is not None:
    #        line += " " + remission_duration[i]
    #    if remission_duration[i] is not None:
    #        line += " " + overall_survival[i]
        print line

if __name__ == '__main__':
    main()

