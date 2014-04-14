'''
Calculates the k-nearest neighbor (kNN) algorithm

'''

import math
import numpy as np
import scipy.io


from kNN import calculate_knn, loading_data
from nkNN import divide_data_folds, return_data_nminusone


__author__ = """Mari Wahl"""


if __name__ == '__main__':
    import sys

    num_k = sys.argv[1] if len(sys.argv) == 2 else 100
    num_n = sys.argv[2] if len(sys.argv) == 3 else 10
    datafile = sys.argv[3] if len(sys.argv) == 4 else 'cvdataset.mat'

    trainingdata, traininglabels, testdata, evaldata, testlabels = loading_data(datafile)

    data_k = []
    for k in range(num_k):  
            a = calculate_knn(trainingdata, traininglabels, k)
            data_k.append(a)
            print('%d	%.3f' %(k, a))

 

    with open('output_train.dat', 'w') as f:
	f.write(data_n)


    print('\n\n#******* STARTING TESTDATA')
    data_k = []
    for k in range(num_k):  
            a = calculate_knn(testdata, testlabels, k)
            data_k.append(a)
            print('%d	%.3f' %(k, a))

 

    with open('output_test.dat', 'w') as f:
	    f.write(data_n)
