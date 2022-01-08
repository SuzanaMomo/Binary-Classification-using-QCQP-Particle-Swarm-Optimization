import numpy as np
from scipy.linalg import eigh
#preprocessing steps done the real datasets for scaling the data
def scale_data(x1,x2):
    mu1 = np.mean(x1,axis=0)
    sigma1 = np.transpose(np.cov(x1.reshape(x1.shape[1],len(x1))))
    mu2 = np.mean(x2,axis=0)
    sigma2 = np.transpose(np.cov(x2.reshape(x2.shape[1],len(x2))))


    p = mu1 - mu2;


    w, v = eigh(sigma1)
    scaled_x1 = [];
    col_1 = np.matmul(x1,p)
    col_2 = np.matmul(x1,v[:,-1])
    for i in range(len(col_1)):
        scaled_x1.append([col_1[i], col_2[i]])
    scaled_x1 = np.array(scaled_x1)

    w, v = eigh(sigma2)
    scaled_x2 = [];
    col_1 = np.matmul(x2,p)
    col_2 = np.matmul(x2,v[:,-1])
    for i in range(len(col_1)):
        scaled_x2.append([col_1[i], col_2[i]])
    scaled_x2 = np.array(scaled_x2)
    return scaled_x1, scaled_x2
   