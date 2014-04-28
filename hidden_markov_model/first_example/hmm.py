import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

# plot setup
matplotlib.rcParams.update({'font.size': 15})


# setting the components of HMM
'''
    P: a fair die is twice as likely as biased die

    A  : The die thrower likes to keep in one state (fair/biased), and the tranisition from 
    1. Fair-> Fair : .95
    2. Fair->Biased: 1-.95=.05
    3. Biased->Biased: .90
    4. Biased->Biased=1-.90=.10

    B  : The fair die is equally likely to produce observations 1 through 6, for the biased die
    Pr(6)=0.5 and Pr(1)=Pr(2)=Pr(3)=Pr(4)=Pr(5)=0.1
'''

P = np.array([2.0/3,1.0/3])
A = np.array([[.95,.05],[.1,.9]])
B = np.array([[1.0/6 for i in range(6)],[.1,.1,.1,.1,.1,.5]])



'''
   Returns next state according to weigted probability array. 
'''
def next_state(weights):
    choice = random.random() * sum(weights)
    for i, w in enumerate(weights):
        choice -= w
        if choice < 0:
            return i



def create_hidden_sequence(P,A,length):
    out=[None]*length
    out[0]=next_state(P)
    for i in range(1,length):
        out[i]=next_state(A[out[i-1]])
    return out

def create_observation_sequence(hidden_sequence,B):
    length=len(hidden_sequence)
    out=[None]*length
    for i in range(length):
        out[i]=next_state(B[hidden_sequence[i]])
    return out

'''Group all contiguous values in tuple.'''
def group(L):
    first = last = L[0]
    for n in L[1:]:
        if n - 1 == last: # Part of the group, bump the end
            last = n
        else: # Not part of the group, yield current group and start a new
            yield first, last
            first = last = n
    yield first, last # Yield the last group

'''Create tuples of the form (start, number_of_continuous values'''
def create_tuple(x):
    return [(a,b-a+1) for (a,b) in x]




if __name__ == '__main__':
	count = 0

	for i in range(1000):
	    count += next_state(P)

	print "Expected number of Fair states:", 1000-count
	print "Expected number of Biased states:", count

	

	hidden=np.array(create_hidden_sequence(P,A,1000))
	observed=np.array(create_observation_sequence(hidden,B))

	#Tuples of form index value, number of continuous values corresponding to Fair State
	indices_hidden_fair=np.where(hidden==0)[0]
	tuples_contiguous_values_fair=list(group(indices_hidden_fair))
	tuples_start_break_fair=create_tuple(tuples_contiguous_values_fair)

	#Tuples of form index value, number of continuous values corresponding to Biased State
	indices_hidden_biased=np.where(hidden==1)[0]
	tuples_contiguous_values_biased=list(group(indices_hidden_biased))
	tuples_start_break_biased=create_tuple(tuples_contiguous_values_biased)

	#Tuples for observations
	observation_tuples=[]
	for i in range(6):
	    observation_tuples.append(create_tuple(group(list(np.where(observed==i)[0]))))

	plt.subplot(2,1,1)
	plt.xlim((0,1000));
	plt.title('Observations');
	for i in range(6):
	    plt.broken_barh(observation_tuples[i],(i+0.5,1),facecolor='k');
	plt.subplot(2,1,2);
	plt.xlim((0,1000));
	plt.title('Hidden States Green:Fair, Red: Biased');
	plt.broken_barh(tuples_start_break_fair,(0,1),facecolor='g');
	plt.broken_barh(tuples_start_break_biased,(0,1),facecolor='r');
	plt.savefig('hmm.png')