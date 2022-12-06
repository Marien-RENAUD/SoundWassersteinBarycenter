from random import random
import warnings
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy as sp
import scipy.signal
import scipy.io.wavfile
import matplotlib.animation as animation
from time import time
from IPython.display import HTML
from tqdm import tqdm
import librosa
import ot as pot
from scipy.interpolate import interp1d
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend

#In order to compute barycenter between sounds

class Wasserstein_barycenter:

    def __init__(self, adresse1, adresse2, method, alpha = 0.5, M3 = np.zeros(1), t_compute = 0, M_f1 = np.zeros(1), M_f2 = np.zeros(1), X1 = np.zeros(1), X2 = np.zeros(1), fs1 = 0, length = 0, Z = np.zeros(0), nfft = 0):
        """
        Initialize a Wasserstein_barycenter object.

        Parameters
        ----------
        self : Wasserstein_barycenter_sound
            objet where we applied this method
        adresse1 : str
            adresse of the source sound
        adresse2 : str
            adresse of the target sound
        method : str
            type of method of compute for the barycenter. Methods Choices are : L2, quantile1D, unbalanced1D, efficient_sinkhorn
        alpha : float
            $\alpha \in [0,1]$ the parameter of the computed barycenter
        M3 : array
            the computed barycenter (spectrogram) between sound adresse1 and adresse 2
        t_compute : float
            compute time
        M_f1 : array
            STFT of the source sound
        M_f2 : array
            STFT of the target sound
        X1 : array
            temporal array of the source sound
        X2 : array
            tamporal array of the target sound
        fs1 : int
            sampling rate
        length : int
            temporal length of the processing sound
        Z : numpy array
            sound to be save (wasserstein barycenter or a sequence of barycenters)
        nfft : int
            the window length for STFT and iSTFT tranforms
        """
        self.adresse1 = adresse1
        self.adresse2 = adresse2
        self.method = method
        self.alpha = alpha
        self.M3 = M3
        self.t_compute = t_compute
        self.M_f1 = M_f1
        self.M_f2 = M_f2
        self.X1 = X1
        self.X2 = X2
        self.fs1 = fs1
        self.length = length
        self.Z = Z
        self.nfft = nfft

    fc = lambda n, p : (np.arange(p)**2/((n-1)**2+(p-1)**2))
    fd = lambda n, p : (np.arange(n)**2/((n-1)**2+(p-1)**2))
        
    def compute(self, p = 1, sound=False, printing=False, alpha = 0.5 , n_it = 10, epsilon = 1e-2, reg = 1.0, nperseg = None, length = 0, method_phase = "adaptive_phase", c=fc, d=fd):
        """
        Function to compute barycenters between the two sound in the adresse self.adresse1 and self.adress2. We use the method of computation self.method. We computed all the barycenter with a regular step.

        Parameters
        ----------
        self : Wasserstein_barycenter_sound
            objet where we applied this method
        p : int
            number of temporal available displacement for mass for the method Efficient Sinkhorn
        sound : boolean
            determine if we heard the two sound source and target before the computation.
        printing : boolean
            determine if we print parameters n, m and the STFT of the source and target sound.
        n_it : int
            number of iteration in the optimal transport computation for the Sinkhorn Sparse method
        epsilon : float
            regularization parameter for Sinkhorn method (Efficient Sinkhorn or unbalanced)
        reg : float
            Marginal relaxation term > 0 for unbalanced1D method
        nperseg : int
            number of point for FFT computation in the STFT computation of the source and target sound
        length : int
            length of the temporal signal which is analyse. The Standard value is the minimum length of the two sound
        method_phase : str
            Use to choose between the sound phase reconstruction method (adaptative phase and Griffin-Lim). 
        c : tensor-like, shape(p)
            temporal cost
        d : tensor-like, shape(n)
            frenquencial cost
        """
        
        adresse1 = self.adresse1
        adresse2 = self.adresse2
        method = self.method
        alpha = self.alpha

        X1, fs1 = librosa.load(adresse1, sr=None)
        X2, fs2 = librosa.load(adresse2, sr=None)

        if fs1 != fs2 :
            raise ValueError(fs1)
        self.fs1 = fs1

        if sound:
            sd.play(np.concatenate([X1,X2]),fs1)

        #Assure that different sounds have the same length
        if length == 0:
            self.length = min(len(X1), len(X2))
        else :
            self.length = length
        X1 = X1[:self.length]
        X2 = X2[:self.length]
        self.X1 = X1
        self.X2 = X2        

        #A standard window of 40 ms 
        if nperseg == None:
            nperseg = 2**(round(np.log2(int(fs1*40e-3))))
            print("nperseg = ",nperseg)
        self.nfft = nperseg
        
        #Compute the STFT
        M_f1 = librosa.stft(X1, n_fft=nperseg)
        self.M_f1 = M_f1
        M1 = np.abs(M_f1)

        if printing:
            plt.figure(figsize=(16,6)); plt.title("Spectrogram of the source sound "+str(adresse1))
            plt.pcolormesh(M1, shading='gouraud')
            plt.show()
        
        F = len(M1)
        T = len(M1[0])
        
        #Compute the STFT
        M_f2 = librosa.stft(X2, n_fft=nperseg)
        self.M_f2 = M_f2
        M2 = np.abs(M_f2)

        if printing:
            plt.figure(figsize=(16,6)); plt.title("Spectrogram of the target sound "+str(adresse2))
            plt.pcolormesh(M2, shading='gouraud')
            plt.show()

            print("T = "+str(T))
            print("F = "+str(F))
        
        
        #Different method of interpolation
        
        if method == "L2":
            t_build = time()
            M3 = (M1 + M2)/2
            t_build = time() - t_build
        
        if method == "quantile1D":
            M3, t_build = barycenter_quantile(M1,M2,alpha)
        
        if method == "unbalanced1D":
            M3, t_build = barycenter_unbalanced(M1, M2, alpha, epsilon, reg)
            
        if method == "efficient_sinkhorn_round":
            M3, t_build = barycenter_sinkhorn_sparse_round(M1, M2, alpha, n_it, epsilon, p, c, d)
            
        if method == "efficient_sinkhorn":
            M3, t_build = efficient_sinkhorn(M1, M2, p, d, c, epsilon, alpha, n_it)
        
        self.M3 = M3
        self.t_build = t_build
        
        #Compute the sound in temporal domain with phase reconstruction   
        if method_phase == "adaptive_phase":
            S = M3 * ((1-alpha)* M_f1 / np.abs(M_f1)  + alpha * M_f2 / np.abs(M_f2))
            Z = librosa.istft(S,length = self.length)
        else :
            Z = librosa.griffinlim(M3,n_fft=self.nfft)
        Z = ((1-alpha)*np.max(X1) + alpha*np.max(X2))*Z/np.max(Z)

        self.Z = Z
        
        
    def save_sound(self, adresse3):
        """
        Function to save the computed sound with the self.method in adresse3.

        Parameters
        ----------
        self : Wasserstein_barycenter_sound
            objet where we applied this method
        adresse3 : str
            adresse where to save the sound created by the algorithm (need to end by .wav)
        """
        Z = self.Z
        fs1 = self.fs1
        sf.write(adresse3, Z, fs1, subtype='PCM_24')

        
###
#Quantile method
###

def inverse_histograms(mu, S, Sinv, method='linear'):
    """
    Given a distribution mu, compute its inverse quantile function

    Parameters
    ----------

    mu     : histogram
    S      : support of the histogram
    Sinv   : support of the quantile function
    method : name of the interpolation method (linear, quadratic, ...)

    Returns
    -------

    cdfa   : the cumulative distribution function
    q_Sinv : the inverse quantile function of the distribution mu

    """

    epsilon = 1e-14
    A = mu>epsilon
    A[-1] = 0
    Sa = S[A]
    
    cdf = np.cumsum(mu)
    cdfa = cdf[A]
    if (cdfa[-1] == 1):
        cdfa[-1] = cdfa[-1] - epsilon

    cdfa = np.append(0, cdfa)
    cdfa = np.append(cdfa, 1)

    if S[0] < 0:
        print('weird for a psd!')
        Sa = np.append(S[0]-1, Sa)
    else:
        # set it to zero in case of PSDs
        Sa = np.append(0, Sa)
    Sa = np.append(Sa, S[-1])

    q = interp1d(cdfa, Sa, kind=method)
    q_Sinv = q(Sinv)
    return cdfa, q_Sinv

def get_barycenter_mult(mus, S, n, weights, method='linear'):
    """

    Compute the Wasserstein barycenter of 1d-distributions

    Parameters
    ----------

    mus    : NxD matrix that contains the D samples of N distributions
    S      : the D support points
    n      : the number of points used for the support of the quantile function
    weights : p*N matrix, weights of each distributions for barycenter computation. p is the number of computed barycenter


    Returns
    -------

    res      : Wasserstein barycenters of distributions mus with weights.
    Finv     : the inverse quantile fuction of distributions mus.
    Finv_bar : inverse quantile function of barycenters.

    """
    #Compute of every distributions quantile function
    N, D = mus.shape
    Finv = np.zeros((N,n))
    Sinv = np.linspace(0, 1, n)
    for i in range(N):
        cdfa, Finv[i,:] = inverse_histograms(mus[i,:], S, Sinv, method)
    #Compute Wasserstein barycenters
    p = len(weights)
    
    res = np.zeros((p,D))

    for i in range(p):
        mask = np.ones(n)[None,:]*weights[i,:,None]*N
        Finv_bar = np.mean(Finv*mask, axis=0)
        Sd = np.append(S[0]-1, S)
        cdf = interp1d(Finv_bar, Sinv, bounds_error=False, fill_value=(0,1),kind=method)
        cdf_S = cdf(S)
        res[i] = cdf_S.copy()
        res[i,1:] = res[i,1:] - res[i,:-1]
    return res, Finv, Finv_bar

def barycenter_quantile(M1,M2,alpha):
    """
    Compute a set of Wasserstein barycenter between STFT. The barycenter is computed column by column. The step for the coefficient of barycenter is constant.

    Parameters
    ----------
    M1 : array-like, shape(F,T)
        source STFT
    M2 : array-like, shape(F,T)
        target STFT
    alpha : float
        $\alpha \in [0,1]$ the parameter of the computed barycenter


    Returns
    -------

    M3 : array-like, shape(F,T)
        Computed barycenter
    t_build : 
        Time to compute the barycenter with the quantile method

    """
    F = len(M1)
    T = len(M1[0])
    
    M1_ = M1.copy()
    M2_ = M2.copy()
    
    S = np.arange(F)
    
    M3 = np.zeros((F,T))
    t_build = time()
    
    weights = np.zeros((1,2))
    weights[0,0] = 1-alpha
    weights[0,1] = alpha
    
    for j in tqdm(range(T)):
        V1 = M1_[:,j] / np.sum(M1_[:,j])
        V2 = M2_[:,j] / np.sum(M2_[:,j])
        mus = np.zeros((2,F))
        mus[0] = V1
        mus[1] = V2
        V3, _, _ = get_barycenter_mult(mus, S, F, weights)
        M3[:,j] = np.array(V3 * (alpha*np.sum(M1_[:,j]) + (1-alpha) * np.sum(M2_[:,j])))
    t_build = time() - t_build
    return M3, t_build


###
#Unbalanced method
###

def barycenter_unbalanced(M1, M2, alpha, epsilon, reg):
    t_build = time()
    F = len(M1)
    T = len(M1[0])
    
    #Loss Matrix
    M = pot.utils.dist0(F)
    M /= M.max()
    
    M3 = np.zeros((F,T))
    for j in tqdm(range(T)):
        V1 = M1[:,j]
        V2 = M2[:,j]
        A = np.vstack((V1, V2)).T
        weights = np.array([1 - alpha, alpha])
        M3[:,j] = pot.unbalanced.barycenter_unbalanced(A, M, epsilon, reg, weights=weights)
    t_build = time() - t_build
    return M3, t_build

###
# Efficient Sinkhorn algorithm with a rounding operation
# We just change the product operation in the precedent algorithm and we never construct the cost K (it's a too large matrice)
###


def Optimal_transport_map_computation_sinkhorn_efficient(F,V1,V2,epsilon,p,n_it,c,d):
    """
    Function to compute three objects which are useful to represent the optimal transport map

    Parameters
    ----------
    F : int
        number of frequence in the STFT
    V1 : array-like, shape(N)
        target distribution
    V2 : array-like, shape(N)
        source distribution
    epsilon : float
        Sinkhorn regularisation coefficient
    p : int
        number of temporal available displacement for mass
    n_it : int
        the number of iteration
    c : array-like, shape(p)
        temporal cost
    d : array-like, shape(F)
        frenquencial cost

    Returns
    -------
    b : array-like, shape(F)
        array of coefficient which contains all the coefficient of the bloc B_0 of the matrix K
    u : array-like, shape(N)
        array use to compute the optimal plan
    v : array-like, shape(N)
        array use to compute the optimal plan
    list_u : list of array, shape(n_it,F)
        list all the computed version of u
    list_v : list of array, shape(n_it,F)
        list all the computed version of v
    """
    #Initialisation of the parameters
    N = len(V1)
    T = N//F
    u = np.ones(len(V1))
    list_u = []
    v = np.zeros(len(V2))
    list_v = []

    #Conctruction of the coefficient of the matrix B_0
    b = np.exp(-d(F,p)/epsilon)

    #Computation of an useful vector to have a fast compute of B_i*x
    t = np.concatenate([np.array([b[0]]),np.flip(b[1:],[0])])
    C_ = np.concatenate([b,t])
    t_ = np.concatenate([np.array([C_[0]]),np.flip(C_[1:],[0])])
    lamb = np.fft.fft(t_)
    
    for _ in tqdm(range(n_it)):
        
        #Compute of K.T@u
        u_ = np.zeros(N)

        #Computation of all matrix product
        list_product_1 = np.zeros((T,F))
        for k in range(T):
            V = np.concatenate([u[k*F:(k+1)*F],np.zeros(F)])
            list_product_1[k] = np.real(np.fft.ifft(lamb*np.fft.fft(V)))[:F]
        for k in range(T):
            #update of u_k
            r = np.zeros(F)
            #Fast compute of B[i]@u[k-i]
            if min(p-1,k)>=1:
                tensor_i = np.arange(1,min(p-1,k)+1)
                r += np.sum(np.exp(-c(F,p)[tensor_i]/epsilon)[:,None]*list_product_1[k-tensor_i,:], axis = 0)
            #Fast compute of B[i]@u[k+i]
            tensor_i2 = np.arange(min(p-1,T-k-1)+1)
            r += np.sum(np.exp(-c(F,p)[tensor_i2]/epsilon)[:,None]*list_product_1[k+tensor_i2,:], axis = 0)
            u_[k*F:(k+1)*F] = r

        v = V2 / u_
        list_v.append(v)

        #Compute of K@v
        v_ = np.zeros(N)

        #Computation of all matrix product
        list_product_2 = np.zeros((T,F))
        for k in range(T):
            V = np.concatenate([v[k*F:(k+1)*F],np.zeros(F)])
            list_product_2[k] = np.real(np.fft.ifft(lamb*np.fft.fft(V)))[:F]

        for k in range(T):
            #update of v_k
            r = np.zeros(F)
            if min(p-1,k)+1>1:
                #Fast compute of B[i]@v[k-i]
                tensor_i = np.arange(1,min(p-1,k)+1)
                r += np.sum(np.exp(-c(F,p)[tensor_i]/epsilon)[:,None]*list_product_2[k-tensor_i,:], axis = 0)
            #Fast compute of B[i]@v[k+i]
            tensor_i2 = np.arange(min(p-1,T-k-1)+1)
            r += np.sum(np.exp(-c(F,p)[tensor_i2]/epsilon)[:,None]*list_product_2[k+tensor_i2,:], axis = 0)
            v_[k*F:(k+1)*F] = r

        u = V1 / v_
        list_u.append(u)
    return b, u, v, list_u, list_v

def arrondie(x):
    if x-int(x)<0.5:
        return int(x)
    else:
        return int(x)+1
    
arrondie = np.vectorize(arrondie)

def barycenter_computation_sinkhorn_efficient_round(F, T, alpha, p, epsilon, B_0, u, v):
    """
    Function to compute the magnitude of the interpolant with a pre-computed optimal transport plan. We use the knowledge that a lot of coefficient 
    are zeros in the K-matrix. It's a fast version but with a big spacial cost.

    Parameters
    ----------
    F : int
        number of frequency
    T : int
        number of frame
    alpha : float
        interpolation coefficient
    p : int
        number of temporal available displacement for mass
    epsilon : float
        regularization parameter
    B_0 : array-like, shape(F,F)
        the first bloc of the matrix K
    u : array-like, shape(N)
        array use to compute the optimal plan
    v : array-like, shape(N)
        array use to compute the optimal plan

    Returns
    -------
    M3 : array-like, shape(F,T)
        the magnitude of the interpolation matrix
    """

    M3 = np.zeros((F,T))

    f = np.arange(F)
    indice = arrondie((1-alpha)*f[:,None] + alpha* f[None,:])
    indice = indice[None,:,:] * np.ones(F)[:,None,None]
    S = f[:,None,None] * np.ones((F,F))[None,:,:]
    mask = (indice == S)

    t = np.arange(p)
    liste_P = np.exp(-t**2/(epsilon*((F-1)**2+(p-1)**2)))[:,None,None] * B_0[None,:,:]

    t3 = 0
    for t1 in tqdm(range(T)):

        T2 = np.arange(max(t1-p+1,0),min(t1+p,T))
        k = np.abs(T2 - t1)
        V = np.reshape(v[None,max(t1-p+1,0)*F:min(t1+p,T)*F],(len(T2),1,F))

        M = u[t1*F:(t1+1)*F,None]*V
        P = liste_P[k] * M

        mask_P = mask[None,:,:,:]*P[:,None,:,:]
        Sum_P = np.sum(mask_P, axis=(2,3))
        
        for t2 in T2:
            t2 = t2.item()
            M3[:,arrondie((1-alpha)*t1+alpha*t2)] += Sum_P[t2-max(t1-p+1,0)]

    return M3

def barycenter_sinkhorn_sparse_round(M1, M2, alpha, n_it, epsilon, p, c, d):
    """
    Function to compute the magnitude a barycenter. We suppose that the matrix of optimal transport 
    have a special sparse structure developpe in our work. We use here a round operation : first we compute the optimal 
    transport map then we compute the barycenter.

    Parameters
    ----------
    M1 : array-like, shape(F,T)
        source STFT
    M2 : array-like, shape(F,T)
        target STFT
    alpha : float
        $\alpha \in [0,1]$ the parameter of the computed barycenter
    epsilon : float
        regularization parameter
    p : int
        number of temporal available displacement for mass
    c : array-like, shape(p)
        temporal cost
    d : array-like, shape(F)
        frenquencial cost

    Returns
    -------
    M3 : array-like, shape(F,T)
        Computed barycenter
    t_build : float
        computation time
    """
    F = len(M1)
    T = len(M1[0])
    N = F*T
    
    V1 = (M1.T).reshape(N)
    V2 = (M2.T).reshape(N)
    t_build = time()
    b, u, v, list_u, list_v = Optimal_transport_map_computation_sinkhorn_efficient(F,V1,V2,epsilon,p,n_it,c,d)

    #Built of the matrix B_0
    f = np.arange(F)
    indice_ = np.abs(f[:,None] - f[None,:])
    B_0 = b[indice_]

    M3 = barycenter_computation_sinkhorn_efficient_round(F, T, alpha, p, epsilon, B_0, u, v)
    t_build = time() - t_build


    return M3, t_build


###
# Efficient Sinkhorn algorithm
# We just change the product operation in the precedent algorithm and we never construct the cost K (it's a too large matrice)
###

def efficient_matrical_product(d,c,epsilon,p,T,F,x):
    """
    Function to compute the multiplication of the sparse matrix $K = e^{\frac{-C}{\epsilon}}$, with $C$ the cost, and the vector $x$.

    Parameters
    ----------
    d : array-like, shape(F)
        frenquencial cost
    c : array-like, shape(p)
        temporal cost    
    epsilon : float
        regularization parameter
    p : int
        number of temporal available displacement for mass
    T : int
        number of frame
    F : int
        number of frequency
    x : array-like, shape(N)
        vector to be multiply by K
    
    Returns
    -------
    x_ : array-like, shape(N)
        the result of the multiplication of K and x
    """
    
    #Conctruction of the coefficient of the matrix exp(-cost/epsilon)
    b = np.exp(-d(F,p)/epsilon)

    #Computation of an useful vector to have a fast compute of B_i*x
    t = np.concatenate(([np.array([b[0]]),np.flip(b[1:],[0])]))
    C_ = np.concatenate(([b,t]))
    t_ = np.concatenate(([np.array([C_[0]]),np.flip(C_[1:],[0])]))
    lamb = np.fft.fft(t_)
        
    #Compute of K@x
    x_ = np.zeros(T*F)

    #Computation of all matrix product
    list_product_1 = np.zeros((T,F))
    for k in range(T):
        V = np.concatenate(([x[k*F:(k+1)*F],np.zeros(F)]))
        list_product_1[k] = np.real(np.fft.ifft(lamb*np.fft.fft(V)))[:F]
    for k in range(T):
        #update of x_k
        r = np.zeros(F)
        #Fast compute of B[i]@x[k-i]
        if min(p-1,k)>=1:
            array_i = np.arange(1,min(p-1,k)+1)
            r += np.sum(np.exp(-c(F,p)[array_i]/epsilon)[:,None]*list_product_1[k-array_i,:], axis = 0)
        #Fast compute of B[i]@u[k+i]
        array_i2 = np.arange(min(p-1,T-k-1)+1)
        r += np.sum(np.exp(-c(F,p)[array_i2]/epsilon)[:,None]*list_product_1[k+array_i2,:], axis = 0)
        x_[k*F:(k+1)*F] = r

    return x_

def geometricMean(a,b):
    """return the  geometric mean of two distributions"""
    return np.exp((np.log(a) + np.log(b))/2)

def geometricBar(alpha, a, b):
    """return the weighted geometric mean of two distributions"""
    return np.exp((1-alpha)*np.log(a) + alpha*np.log(b))

def efficient_sinkhorn(M1, M2, p, d, c, epsilon, alpha, numItermax=300):
    """
    Function to compute the magnitude a barycenter. We suppose that the matrix of optimal transport 
    have a special sparse structure developpe in our work.

    Parameters
    ----------
    M1 : array-like, shape(F,T)
        source STFT
    M2 : array-like, shape(F,T)
        target STFT
    p : int
        number of temporal available displacement for mass
    d : array-like, shape(F)
        frenquencial cost
    c : array-like, shape(p)
        temporal cost
    epsilon : float
        the regularization parameter
    alpha : float
        $\alpha \in [0,1]$ the parameter of the computed barycenter
    numItermax : int
        number of iteration
    
    Returns
    -------
    M3 : array-like, shape(F,T)
        Computed barycenter
    t_build : float
        computation time
    """
    
    t_build = time()
    F = len(M1); T = len(M1[0])
    n1 = np.sum(M1); n2 = np.sum(M2)
    M1 = M1 / n1; M2 = M2 / n2
    V1 = M1.reshape(F*T) ; V2 = M2.reshape(F*T)
    
    UKv1 = efficient_matrical_product(d,c,epsilon,p,T,F,V1)
    UKv1 /= np.sum(UKv1)
    UKv2 = efficient_matrical_product(d,c,epsilon,p,T,F,V2)
    UKv2 /= np.sum(UKv2)
    
    gm = geometricMean(UKv1, UKv2)
        
    u1 = gm / UKv1
    u2 = gm / UKv2
    
    for _ in tqdm(range(numItermax)):

        UKv1 = u1 * efficient_matrical_product(d,c,epsilon,p,T,F, (V1 / efficient_matrical_product(d,c,epsilon,p,T,F,u1)) )

        UKv2 = u2 * efficient_matrical_product(d,c,epsilon,p,T,F, (V2 / efficient_matrical_product(d,c,epsilon,p,T,F,u2)) )

        u1 = (u1 * geometricBar(alpha, UKv1, UKv2)) / UKv1
        u2 = (u2 * geometricBar(alpha, UKv1, UKv2)) / UKv2

            
    V3 = geometricBar(alpha, UKv1, UKv2)
    M3 = V3.reshape((F,T))
    t_build = time()-t_build
    return M3, t_build