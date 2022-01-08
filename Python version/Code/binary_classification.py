import numpy as np
from Algorithms import QCQP_PSO
from generate_data import *
from scaled_data import *
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import time

#illustrates the Binary classification on both the algorithms A and B 
# Dataset is simulated data with 1000 datapoints and 10 particles

a = [1, 1]
# A = np.eye(2)

T = 300

print('For Binary Classification on Random dataset')


dataset = 1

k = 10
x1, x1_cov, mu1, x2, x2_cov, mu2 = generate_data(dataset)
p1, p2 = generate_particles(x1, x2, k)

print()
print('Choose which Algorithm you want to run on:')
print('1 for Original Algorithm A,')
print('2 for Improved Algorithm B,')

algo = int(input('Enter 1 or 2: '))

algorithms = QCQP_PSO()

if algo == 1:
	# find best particle for class 1
	xbg1 = algorithms.algorithmAforProject(x1_cov, T, np.transpose(p1), mu2)

	# find best particle for class 2
	xbg2 = algorithms.algorithmAforProject(x2_cov, T, np.transpose(p2), mu1)
else:
	# find best particle for class 1
	xbg1 = algorithms.algorithmBforProject(x1_cov, T, np.transpose(p1), mu2)

	# find best particle for class 2
	xbg2 = algorithms.algorithmBforProject(x2_cov, T, np.transpose(p2), mu1)

# create hyperlane
if xbg2[1]-xbg1[1] == 0:
	hyper_plane_grad = 1
else:
	hyper_plane_grad = -1*(xbg2[0]-xbg1[0])/(xbg2[1]-xbg1[1])
mid_point = []
mid_point.append((xbg2[0]+xbg1[0])/2)
mid_point.append((xbg2[1]+xbg1[1])/2) 
const = mid_point[1] - hyper_plane_grad * mid_point[0]
# hyper_plane_x = np.arange(min(x1[:,0])-20,max(x1[:,0])+20,0.2)
hyper_plane_x = np.arange(-10,15,0.2)
hyper_plane = []
for x in hyper_plane_x:
	hyper_plane.append(hyper_plane_grad * x + const)


plt.plot(x1[:,0],x1[:,1],'.', color='yellow', label='class Y')
plt.plot(x2[:,0],x2[:,1],'.', color='pink', label='class X')
plt.plot(p1[:,0],p1[:,1],'.', color='black', label='Particles')
plt.plot(p2[:,0],p2[:,1],'.', color='black')
plt.plot(xbg1[0],xbg1[1],'x', color='red', label='Best Particles')
plt.plot(xbg2[0],xbg2[1],'x', color='red')
plt.plot(hyper_plane_x,hyper_plane, label='Hyperplane')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
p1, p2 = generate_particles(x1, x2, k)
# plt.title("Binary Classification using Algo B")
plt.show()
