import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as may
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import time
start = time.time()

#Generate test input#
S = list(range(32)) # number of series
N = 512
dt = 1 # timestep
t = np.arange(0.,N,dt) # time domain
t=t/N
n = len(t)
A = np.array([2,3,5]) # amplitudes
w = np.array([120.,230.,350.]) # frequencies
B = np.array([.5,.8,.7]) # damping coeffs
dw = np.array([0.,1.,-1/2]) # changes in frequencies

#Create matrix of function(series)
f = np.zeros((len(S),n),dtype="complex_")
for i in range(len(S)):
    for k in range(3):
        f[i] = np.add(f[i], A[k]*np.e**((2*np.pi*1j*(w[k]+i*dw[k]+1j*B[k])*t)))
        # f[i] = np.add(f[i],.5*np.random.randn(n))
# ftest = np.zeros((len(S),n),dtype="complex_")
# for i in range(len(S)):
#     for k in range(1):
#         ftest[i] = np.add(ftest[i], A[k]*np.e**((2*np.pi*1j*(w[k]+i*dw[k]+1j*B[k])*t)))

# FData = np.fft.fft(f)
# plt.figure()
# ax = plt.subplot(111)
# for i in range(int(len(S)/2)):
#     ax.plot(t,np.real(FData[2*i+1]), label='%f'%i)


#Functions to fit later
def Lorentz(x, A, V, B):
    return np.real(A*B/(B - 1j*(V - x)))
    # return np.abs(1/(B - 1j*(V - x)))

def Lorentzcost(B, dictionary):
    multiplycost = 1
    cost = dictionary['cut'] - Lorentz(dictionary['X'], dictionary['RA'], dictionary['V'], B)
    norm = np.linalg.norm(cost)
    if np.any(cost<0):
        multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
    return norm*multiplycost

def helper(V, A):
    def Lorentzhelp(x, B):
        return np.real(A*B/(B - 1j*(V - x)))
    return Lorentzhelp

#Radon Transform
def Radon(f, dwmin, dwmax, ddw):
    s = np.shape(f)[0]
    n = np.shape(f)[1]
    
    #Phase correction
    DW = np.arange(dwmin,dwmax,ddw) # domain of rates of change
    p = np.zeros((len(DW),s,n),dtype="complex64")
    a = -2*np.pi*1j*t
    for i in range(len(DW)):
        b = a*DW[i]
        for k in range(s):
            p[i][k] = f[k]*np.e**(b*k)
    
    #"Diagonal" summation
    Pr = np.zeros((len(DW),n),dtype="complex64")
    for i in range(len(DW)):
        for j in range(s):
            Pr[i] += p[i][j]
    
    freq = np.arange(n)
    
    #Fourier Transform
    phat = np.zeros((len(DW),n),dtype="complex64")
    PR = np.zeros((len(DW),n))
    phat = np.fft.fft(Pr)
    PR = np.abs(phat)
    
    # Plotting
    may.figure()
    X, Y = np.mgrid[0:len(freq), 0:len(DW)]
    may.surf(X, Y, PR, warp_scale=np.min(np.shape(PR))/np.max(PR))
    
    #Peak Picker
    result = np.where(PR == np.amax(PR)) # coordinates of global maxima
    cords1, cords2 = result[0][0], result[1][0]
    dw1, w1 = DW[cords1], freq[cords2] # frequency and rate of change of highest peak
    print("rate of change: " + str(dw1), "frequency: " + str(w1))
    #*(ppmmin+(abs(ppmmin-ppmmax)-b2)-ppmmax+b1/(1000*n))
    return PR, cords1, w1, dw1, phat

def findbeta(Slice, V, RA):  
    rinterval = 30
    cut = Slice[V-rinterval:V+rinterval]
    X = np.linspace(V-rinterval,V+rinterval-1,len(cut))
    
    #Initial parameters fit
    popt,_ = curve_fit(Lorentz, X, cut, bounds=((RA*0.995,V*0.95,0), (RA*1.005,V*1.05,15)))
    fit = Lorentz(X, *popt)
    RAfit, Vfit, Bini = popt[0], popt[1], popt[2]
    print(RAfit, Vfit, Bini)
    
    dictionary = {'RA':RAfit,'V':Vfit,'cut':cut,'X':X}
    
    #Optimizing optimization algorithms
    Bopt = minimize(Lorentzcost, x0=(Bini), args=(dictionary), method='Nelder-Mead')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(X,cut)
    ax.plot(X, fit, 'r-', label='initial fit')
    # plt.vlines([V], 0, RA) # to check if alignes correctly
    ax.plot(X, Lorentz(X, RAfit, Vfit, Bopt.x[0]), label='final fit')
    ax.legend()
    
    print("Beta found: %f" % Bopt.x[0])
    return Bopt.x[0]


# def findbeta(Slice, V, RA): #V - signal frequency, RA - amplitude in Radon spectrum
#     rinterval = 100
#     cut = Slice[V-rinterval:V+rinterval]
#     tofit = helper(V, RA)
#     X = np.linspace(V-rinterval,V+rinterval-1,len(cut))
#     popt, pcov = curve_fit(tofit, X, cut, bounds=((0), (15)))
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(X,cut)
#     ax.plot(X, tofit(X, *popt), 'r-', label='fit')
#     # plt.vlines([V], 0, RA) # to check if alignes correctly
#     ax.legend()
#     plt.figure()
#     plt.plot(t,Slice)
#     print("Beta found: %f" % popt[0])
#     return popt[0]

#def findbetatest

def findA(f, w, dw, B):
    A = 0
    C = 0
    for i in range(0,np.shape(f)[0]):
        for j in range(0,np.shape(f)[1]):
            A += np.e**((2*np.pi*(-1j)*(w+i*dw+(-1j)*B)*j/np.shape(f)[1]))*f[i][j]
            C += np.e**(2*np.pi*(-2*B*j/np.shape(f)[1]))
    return A/C

def findA2(f, w, dw, B):
    '''
    1) generowanie sygnału i transformacja Radona, analog linijki 109
    '''
    return

def subtractpeak(f, w, dw, B, A):
    ffit = np.zeros((np.shape(f)[0],np.shape(f)[1]),dtype="complex64")
    for i in range(np.shape(f)[0]):
        ffit[i] = np.add(ffit[i], A*np.e**((2*np.pi*1j*(w+i*dw+1j*B)*t)))
    
    # Radon(ffit,dwmin,dwmax,ddw)
    fnext = f-ffit
    # subtracted, cords1, w1, dw1 = Radon(fnext,-2,2,.01)
    print("Peak subtracted at:" + "\n" + "w = {:.2f}, dw = {:.2f}, B = {:.2f}, A = {:.2f}".format(w, dw, B, A))
    return fnext

dwmin = -2
dwmax = 2
ddw = .01
#array for peaks found
Peaks = np.empty((0,4), dtype=float)

def dothething(f,dwmin,dwmax,ddw, Peaks):
    PR, cords, w, dw, phat = Radon(f,dwmin,dwmax,ddw)
    slicereal = np.real(phat[cords])
    RA = np.max(np.real(phat[cords])) #Radon amplitude
    B = findbeta(slicereal, w, RA)
    # slice1 = PR[cords]
    # RA = np.max(PR[cords]) #Radon amplitude
    # B = findbeta(slice1, w, RA)
    A = findA(f, w, dw, B)
    fnext = subtractpeak(f, w, dw, B, A)
    Peaks = np.append(Peaks, [[w,dw,B,A]], axis=0)
    return fnext, Peaks

fnext1, Peaks = dothething(f,dwmin,dwmax,ddw, Peaks)
fnext2, Peaks = dothething(fnext1,dwmin,dwmax,ddw, Peaks)
fnext3, Peaks = dothething(fnext2,dwmin,dwmax,ddw, Peaks)

PR4, cords4, w4, dw4, phat4 = Radon(fnext3,dwmin,dwmax,ddw)

print("Summary: ")
print(*np.round(np.real(Peaks),3), sep='\n')

print('It took', time.time()-start, 'seconds.')