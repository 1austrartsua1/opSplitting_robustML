
'''

Code that defines and implements the distributionally robust logistic regression problem found in our paper: 
"Stochastic Projective Splitting: Solving Saddle-Point Problems with Multiple Regularizers"
https://arxiv.org/pdf/2106.13067.pdf
'''

import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import svds

class DR_s_LR:
    def __init__(self,X,y,delta,kappa,l1coef=None):
        self.X = X # regression matrix
        self.y = y # labels
        self.delta = delta # wasserstein ball radius
        self.kappa = kappa # wasserstein metric constant
        self.l1coef = l1coef # l1 regularizer constant
        self.n,self.d = X.shape
        self.gamma_start = self.d+1
        self.num_var = 1 + self.d + self.n
        self.last_batch = []

    def components(self,z):
        lam = z[0]
        beta = z[1:self.gamma_start]
        gamma = z[self.gamma_start:]
        return lam,beta,gamma

    def inv_components(self,lam,beta,gamma):
        return np.concatenate((np.array([lam]),beta,gamma))

    def getObjective(self,z):
        '''
        Phi(t) = log(1+e^{-2t})+t
        '''
        def Phi(t):
            return np.log(1+np.exp(-2*t)) + t

        lam, beta, gamma = self.components(z)
        out = lam*(self.delta-self.kappa)

        Xbeta = self.X.dot(beta)
        out += (1/self.n)*sum(Phi(Xbeta))

        out += (1/self.n)*sum(gamma*(Xbeta*self.y - lam*self.kappa))

        return out


    def getStochasticUpdate(self,batchsz,z,reuseBatch):
        grad = np.zeros(1+self.d+self.n)

        if batchsz == "full":
            batch = np.arange(self.n)
            batchsz = self.n
        elif (reuseBatch==False) or (len(self.last_batch) == 0):
            batch = np.random.choice(np.arange(self.n),batchsz,replace=False)
            self.last_batch = batch
        else:
            # reuseBatch=True and last_batch exists
            batch = self.last_batch



        lam,beta,gamma = self.components(z)

        grad[0] = self.delta - self.kappa*(1+(1/batchsz)*sum(gamma[batch]))
        XbatchBeta = self.X[batch].dot(beta)
        grad[1:self.gamma_start] =  (1/batchsz)*self.X[batch].T.dot(expit(XbatchBeta))
        grad[1:self.gamma_start] += (1/batchsz)*self.X[batch].T.dot(self.y[batch]*gamma[batch])

        gamma_grad = grad[self.gamma_start:]
        gamma_grad[batch] = -(1/batchsz)*(self.y[batch]*XbatchBeta - lam*self.kappa)


        return grad

    def project_conePlusBall(self,z):
        return self.project_cone(self.project_Linf_ball(z))

    def project_cone(self,z):
        lam,beta,gamma = self.components(z)
        #L is 2 for the Log reg loss
        alpha = 1.0/2.0
        betanorm = np.linalg.norm(beta,2)
        if betanorm <= alpha*lam:
            return z

        coef = (lam + betanorm)/2.0
        return self.inv_components(coef, coef*alpha*beta/betanorm, gamma)


    def project_Linf_ball(self,z):
        lam,beta,gamma = self.components(z)

        ones = np.ones(gamma.shape)
        gamma_proj = (gamma<=1.0)*(gamma>=-1.0)*gamma + (gamma>1.0)*ones - (gamma<-1.0)*ones

        return self.inv_components(lam, beta, gamma_proj)


    def prox_L1(self,z,tau):
        #tau*l1coef*ell_1(beta)
        lam, beta, gamma = self.components(z)
        beta = proxL1(beta,tau,self.l1coef)
        return self.inv_components(lam,beta,gamma)

    def get_largest_sv(self):
        # return Lipschitz constant of the vector field B(z)
        sigma_mx_of_X = svds(self.X,k=1,return_singular_vectors=False)[0]
        m = self.X.shape[0]
        L = (self.kappa**2 + sigma_mx_of_X**2)/m + 2*sigma_mx_of_X**2/(m**2)
        return L









def proxL1(a,tau, coef):
    '''
    proximal operator for lambda||x||_1 with stepsize tau
    aka soft thresholding operator
    '''
    taucoef = tau * coef
    x = (a> taucoef)*(a-taucoef)
    x+= (a<-taucoef)*(a+taucoef)
    return x