import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as may
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sys import exit

import time
start = time.time()

##Azaron
datareal = r"C:\Daniel\Programowanie\Radon_transform\sample_data\azaron_new\fid_r.txt"
dataimag = r"C:\Daniel\Programowanie\Radon_transform\sample_data\azaron_new\fid_i.txt"
S = 20 #Number of series, lines
ppmmin = -1.97
ppmmax = 13.97
b1, b2 = 3.75, 4 #boundaries of frequencies of interest (from higher to lower)

##Osocze
# datareal = r"C:\Daniel\Programowanie\Radon_transform\sample_data\osocze\fid_r.txt"
# dataimag = r"C:\Daniel\Programowanie\Radon_transform\sample_data\osocze\fid_i.txt"
# S = 21 #Number of series, lines

with open(datareal) as file:
    datar = np.array([line.split() for line in file], dtype='float_')

with open(dataimag) as file:
    datai = np.array([line.split() for line in file], dtype='float_')

Data = np.add(datar,1j*datai,dtype='complex64')

n = len(Data[0])

Data = np.conj(Data)
FData = np.fft.fft(Data)
FData = np.fft.fftshift(FData,axes=1)
plt.figure()
ax = plt.subplot(111)
ax.plot(np.linspace(ppmmax,ppmmin,n),np.real(FData[0]))
# ax.plot(np.linspace(ppmmax,ppmmin,n),np.real(FData[-1]))
ax.invert_xaxis()

chop1, chop2 = int(np.shape(Data)[1]*abs(ppmmax-b2)/abs(ppmmin-ppmmax)), int(np.shape(Data)[1]*abs(ppmmax-b1)/abs(ppmmin-ppmmax))
FData = FData[:,chop1:chop2]
n = np.shape(FData)[1]
Data = np.fft.ifft(FData,n)

FData = np.fft.fft(Data)
plt.figure()
ax = plt.subplot(111)
ax.plot(np.linspace(b2,b1,n),np.real(FData[0]))
ax.invert_xaxis()

t = np.linspace(0.,1,n)

plt.figure()
ax = plt.subplot(111)
for i in range(S):
    ax.plot(np.linspace(b2,b1,n),np.real(FData[i]), label='%f'%i)

# exit()

#Functions to fit later
def Lorentz(x, A, V, B):
    return np.real(A*B/(B - 1j*(V - x)))

def Lorentzcost(x0, dictionary):
    multiplycost = 1
    cost = dictionary['cut'] - Lorentz(dictionary['X'], x0[0], x0[1], x0[2])
    norm = np.linalg.norm(cost)
    if np.any(cost<0):
        multiplycost *= (1+100*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
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
    a = 2*np.pi*1j*t
    for i in range(len(DW)):
        b = a*DW[i]
        for k in range(s):
            p[i][k] = f[k]*np.e**(b*k)
    
    # #Debugging
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(f)[0]),label='Pierwsza seria')
    ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(f)[-1]),label='Ostatnia seria')
    ax1.invert_xaxis()
    ax1.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(S):
        Y = np.ones(n)*i
        ax.plot(np.arange(n),Y,np.real(np.fft.fft(f)[i]), alpha = 1, zorder = -i)
    ax.set_ylabel('series')
    ax.set_xlabel('frequency')
    ax.set_zlabel('power')
    
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

# def findbeta(Slice, V, RA): #V - signal frequency, RA - amplitude in Radon spectrum
#     rinterval = 30
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
#     print(RA,V,popt[0])
#     # print("Beta found: %f" % popt[0])
#     return popt[0]

def findbeta(Slice, V, RA):  
    rinterval = 20
    cut = Slice[V-rinterval:V+rinterval]
    X = np.linspace(V-rinterval,V+rinterval-1,len(cut))
    
    #Initial parameters fit
    popt,_ = curve_fit(Lorentz, X, cut, bounds=((RA*0.995,V*0.95,0), (RA*1.005,V*1.05,15)))
    fit = Lorentz(X, *popt)
    RAini, Vini, Bini = popt[0], popt[1], popt[2]
    
    dictionary = {'cut':cut,'X':X}
    x0 = [RAini, Vini, Bini]
    
    #Optimizing optimization algorithms
    OptParams = minimize(Lorentzcost, x0, args=(dictionary), method='COBYLA')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(X,cut)
    ax.plot(X, fit, 'r-', label='initial fit')
    # plt.vlines([V], 0, RA) # to check if alignes correctly
    ax.plot(X, Lorentz(X, OptParams.x[0], OptParams.x[1], OptParams.x[2]), label='final fit')
    ax.legend()
    
    fig = plt.figure()
    plt.plot(t,Slice, label='przekrój')
    plt.legend()
    
    print("Beta found: %f" % OptParams.x[2])
    return OptParams.x[2]

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


dwmin = -4
dwmax = 4
ddw = .01

#array for peaks found
Peaks = np.empty((0,4), dtype=float)

def dothething(f,dwmin,dwmax,ddw, Peaks):
    PR, cords, w, dw, phat = Radon(f,dwmin,dwmax,ddw)
    slicereal = np.real(phat[cords])
    RA = np.max(np.real(phat[cords])) #Radon amplitude
    B = findbeta(slicereal, w, RA)
    A = findA(f, w, dw, B)
    fnext = subtractpeak(f, w, dw, B, A)
    Peaks = np.append(Peaks, [[(b2-w/n*abs(b2-b1),'ppm'),(dw*abs(b2-b1)*1000/n,'ppb/K'),B,A]], axis=0)
    #Plot fist series
    FData = np.fft.fft(f)
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(np.linspace(b2,b1,n),np.real(FData[0]),label='Seria 0')
    ax.invert_xaxis()
    ax.legend()
    
    print(b2-w*abs(b2-b1)/n)
    print(b2-(w-S*dw)*abs(b2-b1)/n)
    # print(abs(b2-b1)/n)
    # plt.figure()
    # ax1 = plt.subplot(111)
    # ax1.plot(np.linspace(0,n,n),np.real(phat[int(cords/2)]),label='przekrój')
    # ax1.legend()
    return fnext, Peaks

fnext1, Peaks = dothething(Data,dwmin,dwmax,ddw, Peaks)
fnext2, Peaks = dothething(fnext1,dwmin,dwmax,ddw, Peaks)
# fnext3, Peaks = dothething(fnext2,dwmin,dwmax,ddw, Peaks)
# fnext4, Peaks = dothething(fnext3,dwmin,dwmax,ddw, Peaks)
# fnext5, Peaks = dothething(fnext4,dwmin,dwmax,ddw, Peaks)
# PR4, cordslast, wlast, dwlast, phatlast = Radon(fnext5,dwmin,dwmax,ddw)

print("Summary: ")
print(Peaks, sep='\n') #*np.round(np.real(Peaks),9), sep='\n')

print('It took', time.time()-start, 'seconds.')