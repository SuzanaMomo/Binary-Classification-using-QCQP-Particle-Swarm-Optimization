import numpy as np
from Algorithms import QCQP_PSO
from generate_data import *
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import time
#mainly records the runtimes based on the datsets chosen
#records each runtime in k_times array for Original Algorithm (algorithm A) (20,000 simulated dataset) input growing from 10 to 1000 (testcases)
#records each runtime in logk_times array for Improved Algorithm (algorithm B)(20,000 simulated dataset) input growing from 10 to 1000 (testcases)
# same for real datasets but with varying inputs

def main():
	print('Choose the Dataset you want to run on:')
	print('1 for random dataset,')
	print('2 for Iris dataset,')
	print('3 for Pima dataset,')
	print('4 for Balance dataset,')
	print('5 for Bupa dataset,')
	print('6 for haberman dataset,')

	dataset = int(input('Enter 1 or 2 or 3 or 4 or 5 or 6: '))
	
	run_plot_simulation(dataset)
	

def run_plot_simulation(dataset):
	T = 300 # Number of Iterations for the Algorithm 
	k_times = [] # stores times for Original Algorithm (algorithm A)
	logk_times = [] # stores times for Improved Algorithm (algorithm B)

	x1, x1_cov, mu1, x2, x2_cov, mu2 = generate_data(dataset) # Generates data
	min_no_particles = min(len(x1),len(x2))
	
	if dataset == 1:
		# for random data no of particles increses from 10 to 1000
		K = np.arange(10,1000,1) 
	else:
		# for real data no of particles increses from 10 to lowest number of instances in a class
		K = np.arange(10,min_no_particles,1)

	algorithms = QCQP_PSO()

	for k in K:
		p1, p2 = generate_particles(x1, x2, k) # generates particles

		start = time.time() # records time for Algorithm A
		
		# find best particle for class 1
		xbg1 = algorithms.algorithmAforProject(x1_cov, T, np.transpose(p1), mu2)
		# find best particle for class 2
		xbg2 = algorithms.algorithmAforProject(x2_cov, T, np.transpose(p2), mu1)

		elapsed_time = time.time() - start
		k_times.append(elapsed_time)
		
		start = time.time() # records time for Algorithm B

		# find best particle for class 1
		xbg1 = algorithms.algorithmBforProject(x1_cov, T, np.transpose(p1), mu2)
		# find best particle for class 2
		xbg2 = algorithms.algorithmBforProject(x2_cov, T, np.transpose(p2), mu1)

		elapsed_time = time.time() - start
		logk_times.append(elapsed_time)

	# Plots Algorithm A's and B's timing
	plt.plot(K,k_times, label='Algo A')
	plt.plot(K,logk_times, label='Algo B')
	plt.xlabel('n')
	plt.ylabel('time (sec)')
	plt.legend(loc="upper left")
	plt.show()

if __name__ == "__main__":
	main()