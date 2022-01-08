import pandas as pd 
import numpy as np
from scaled_data import *
#generating data from .csv files of the real datasets

def generate_data(identifier):
	if identifier == 1:
		print('Random')
		return random_data()
	elif identifier == 2:
		print('iris')
		return iris()
	elif identifier == 3:
		print('pima')
		return pima()
	elif identifier == 4:
		print('balance')
		return balance()
	elif identifier == 5:
		print('bupa')
		return bupa()
	elif identifier == 6:
		print('haberman')
		return haberman()

def generate_particles(x1, x2, k):
	p1 = x1[np.random.choice(len(x1),k,replace=False)]
	p2 = x2[np.random.choice(len(x2),k,replace=False)]
	return p1, p2


def haberman():
	df = pd.read_csv('haberman.csv')
	x1 = []
	x2 = []
	for index, row in df.iterrows():
		if row['class'] == 1:
			x1.append([row['f1'], row['f2'], row['f3']])
		elif row['class'] == 2:
			x2.append([row['f1'], row['f2'], row['f3']])
	
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1, x2 = scale_data(x1,x2)
	
	x1_cov = np.transpose(np.cov(x1.reshape(2,len(x1))))
	mu1 = np.mean(x1,axis=0)

	x2_cov = np.transpose(np.cov(x2.reshape(2,len(x2))))
	mu2 = np.mean(x2,axis=0)

	return x1, x1_cov, mu1, x2, x2_cov, mu2	

def bupa():
	df = pd.read_csv('bupa.csv')
	x1 = []
	x2 = []
	for index, row in df.iterrows():
		if row['class'] == 1:
			x1.append([row['f1'], row['f2'], row['f3'], row['f4'], row['f5'], row['f6']])
		elif row['class'] == 2:
			x2.append([row['f1'], row['f2'], row['f3'], row['f4'], row['f5'], row['f6']])
	
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1, x2 = scale_data(x1,x2)
	
	x1_cov = np.transpose(np.cov(x1.reshape(2,len(x1))))
	mu1 = np.mean(x1,axis=0)

	x2_cov = np.transpose(np.cov(x2.reshape(2,len(x2))))
	mu2 = np.mean(x2,axis=0)

	return x1, x1_cov, mu1, x2, x2_cov, mu2	

def balance():
	df = pd.read_csv('balance-scale.csv')
	x1 = []
	x2 = []
	for index, row in df.iterrows():
		if row['class'] == 'R':
			x1.append([row['f1'], row['f2'], row['f3'], row['f4']])
		elif row['class'] == 'L':
			x2.append([row['f1'], row['f2'], row['f3'], row['f4']])
	
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1, x2 = scale_data(x1,x2)
	
	x1_cov = np.transpose(np.cov(x1.reshape(2,len(x1))))
	mu1 = np.mean(x1,axis=0)

	x2_cov = np.transpose(np.cov(x2.reshape(2,len(x2))))
	mu2 = np.mean(x2,axis=0)

	return x1, x1_cov, mu1, x2, x2_cov, mu2	

def pima():
	df = pd.read_csv('pima.csv')
	x1 = []
	x2 = []
	for index, row in df.iterrows():
		if row['class'] == 0:
			x1.append([row['f1'], row['f2'], row['f3'], row['f4'], row['f5'], row['f6'], row['f7'], row['f8']])
		else:
			x2.append([row['f1'], row['f2'], row['f3'], row['f4'], row['f5'], row['f6'], row['f7'], row['f8']])
	
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1, x2 = scale_data(x1,x2)
	
	x1_cov = np.transpose(np.cov(x1.reshape(2,len(x1))))
	mu1 = np.mean(x1,axis=0)

	x2_cov = np.transpose(np.cov(x2.reshape(2,len(x2))))
	mu2 = np.mean(x2,axis=0)

	return x1, x1_cov, mu1, x2, x2_cov, mu2	

def iris():
	df = pd.read_csv('iris.csv')
	x1 = []
	x2 = []
	for index, row in df.iterrows():
		if row['class'] == 'Iris-virginica' or row['class'] == 'Iris-versicolor':
			x1.append([row['sepal_length'], row['sepal_width'], row['petal_length'], row['petal_width']])
		else:
			x2.append([row['sepal_length'], row['sepal_width'], row['petal_length'], row['petal_width']]) 
	
	x1 = np.array(x1)
	x2 = np.array(x2)
	x1, x2 = scale_data(x1,x2)

	x1_cov = np.transpose(np.cov(x1.reshape(2,len(x1))))
	mu1 = np.mean(x1,axis=0)

	x2_cov = np.transpose(np.cov(x2.reshape(2,len(x2))))
	mu2 = np.mean(x2,axis=0)

	return x1, x1_cov, mu1, x2, x2_cov, mu2	
	
def random_data():
	sample = 20000
	# generate random dataset for class 1 
	mu1 = [8, 0]
	sigma1 = [[2, 1],[1, 2]]
	x1 = np.random.multivariate_normal(mu1,sigma1,sample)
	x1_cov = np.transpose(np.cov(x1.reshape(2,sample)))


	# generate random dataset for class 2
	mu2 = [0, 8]
	sigma2 = [[2, 1],[1, 2]]
	x2 = np.random.multivariate_normal(mu2,sigma2,sample)
	x2_cov = np.transpose(np.cov(x2.reshape(2,sample)))

	return x1, x1_cov, mu1, x2, x2_cov, mu2	
