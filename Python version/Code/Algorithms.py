import numpy as np
from scipy.stats import norm, chi2

class QCQP_PSO:

    def f(self, a, x):
        """
        Objective function
        """
        return np.linalg.norm(a-x)

    def algorithmAforProject(self, P, T, X, a):
        """
        Algorithm A
        Finds the best particle that is closest to point 'a'
        Parameters:
        P : Covariance Matrix
        T : no of iterations
        X : K no of particles
        a : Fixed point
        """
        # initialize all variables
        shape = X.shape
        V = np.zeros(shape, dtype=float)
        xb = np.zeros(shape, dtype=float)
        xbg = X[:,0]
        self.f(a, X[:,1])
        for i in range(shape[1]):
            xb[:,i] = X[:,i];
            if self.f(a,xbg) > self.f(a,xb[:,i]):
                xbg = xb[:,i]
        w = 0.05
        c = [0.05, 0.05, 0.05, 0.2]
        t = 0

        while t < T:
            # iteration loop 
            t += 1
            r = np.random.rand(4);
            for i in range(shape[1]):
                # calculate velocity for each particle
                V[:,i] = w*V[:,i] + c[0]*r[0]*(xb[:,i] - X[:,i]) + c[1]*r[1]*(xbg - X[:,i]) + c[2]*r[2]*(a - X[:,i]) + c[3]*r[3]
            
            for i in range(shape[1]):
                xx = X[:,i] + V[:,i]
                if np.matmul(np.matmul(np.transpose(xx),P),xx) <= 1:
                    # update position ofr particle if contraint is met
                    X[:,i] = xx
                
                if self.f(a,xb[:,i])> self.f(a,X[:,i]):
                    # update best position for every particle
                    xb[:,i] = X[:,i]

                if self.f(a,xbg) > self.f(a,xb[:,i]):
                    # find the global best
                    xbg = xb[:,i]

        return xbg # return global best

    def algorithmBforProject(self, P, T, X, a):
        """
        Algorithm A
        Finds the best particle that is closest to point 'a'
        Parameters:
        P : Covariance Matrix
        T : no of iterations
        X : K no of particles
        a : Fixed point
        """
        # initialize all variables
        shape = X.shape
        V = np.zeros(shape, dtype=float)
        xb = np.zeros(shape, dtype=float)
        xbg = X[:,0]
        self.f(a, X[:,1])
        for i in range(shape[1]):
            xb[:,i] = X[:,i];
            if self.f(a,xbg) > self.f(a,xb[:,i]):
                xbg = xb[:,i]
        w = 0.05
        c = [0.05, 0.05, 0.05, 0.2]
        t = 0

        while t < T:
            # iteration loop 
            t += 1
            f_x = []
            r = np.random.rand(4);
            for i in range(shape[1]):
                # calculate velocity for each particle
                V[:,i] = w*V[:,i] + c[0]*r[0]*(xb[:,i] - X[:,i]) + c[1]*r[1]*(xbg - X[:,i]) + c[2]*r[2]*(a - X[:,i]) + c[3]*r[3]
            
            for i in range(shape[1]):
                xx = X[:,i] + V[:,i]
                
                if np.matmul(np.matmul(np.transpose(xx),P),xx) <= 1:
                    # update position ofr particle if contraint is met
                    X[:,i] = xx

                if self.f(a,xb[:,i])> self.f(a,X[:,i]):
                    # update best position for every particle
                    xb[:,i] = X[:,i]

            if shape[1] >= 2:
                # if there is more than two particle left, then half the no of particles
                for i in range(shape[1]):
                    # find objective value of each particle
                    f_x.append(self.f(a,xb[:,i]))

                # sort particle according to objective value
                f_x = np.array(f_x)
                X_f_x = np.vstack((X,f_x))
                X_f_x = X_f_x[:,X_f_x[2].argsort()]
                X = np.delete(X_f_x,2,0)

                xb_f_x = np.vstack((xb,f_x))
                xb_f_x = xb_f_x[:,X_f_x[2].argsort()]
                xb = np.delete(xb_f_x,2,0)

                # remove half of the particles
                xb = np.delete(xb,list(range(int(shape[1]/2)+1,shape[1])),1)            
                X = np.delete(X,list(range(int(shape[1]/2)+1,shape[1])),1)        
                shape = X.shape

            for i in range(shape[1]):
                # find the global best
                if self.f(a,xbg) > self.f(a,xb[:,i]):
                    xbg = xb[:,i]

        return xbg # return global best