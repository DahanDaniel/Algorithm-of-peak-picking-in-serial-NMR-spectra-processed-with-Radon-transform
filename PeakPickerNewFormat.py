import numpy as np
import csv
import matplotlib.pyplot as plt
import mayavi.mlab as may
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sys import exit

iterations = int(input('How many iterations?\n'))

import time
start = time.time()

# Import data # Alanina1
dataspectrum = r"C:\Daniel\Programowanie\Radon_transform\sample_data\Osocze_wspolczynniki_temperaturowe\Alanina1.csv"
Dataspectrum = np.array(list(csv.reader(open(dataspectrum), delimiter="\t")))

# Extract scale in ppm
Scale = Dataspectrum[1:,0]
Scale = Scale.astype('float64')
b1, b2 = float(Scale[0]), float(Scale[-1]) # b1 - smallest, b2 - biggest

# Create data matrix (M[number of series][number of spectrum points])
Dataspectrum = np.delete(np.delete(np.delete(Dataspectrum,0,0),-1,1),0,1)
Dataspectrum = Dataspectrum.astype('float64')

Dataspectrum = Dataspectrum.transpose()

FData = hilbert(Dataspectrum,axis=1)
FData = np.conj(FData)
n = np.shape(FData)[1]
S = np.shape(FData)[0]

Scale = np.linspace(b1,b2,n) # linearize scale

# # Data = np.conj(Data)
# FData = np.fft.fft(Data,axis=-1)
# FData = np.fft.fftshift(FData,axes=-1)
plt.figure()
ax = plt.subplot(111)
ax.plot(Scale,np.real(FData[0]),label='1')
ax.plot(Scale,np.real(FData[-1]),label='-1')
# ax.plot(np.linspace(ppmmax,ppmmin,n),np.real(FData[-1]))
plt.title('Full spectrum')
ax.legend()
ax.invert_xaxis()

# # Cut out the area of interest
# chop1, chop2 = int(np.shape(Data)[1]*abs(ppmmax-b2)/abs(ppmmin-ppmmax)), int(np.shape(Data)[1]*abs(ppmmax-b1)/abs(ppmmin-ppmmax))
# FData = FData[:,chop1:chop2]
# n = np.shape(FData)[1]

Data = np.fft.ifft(FData,n,axis=1) # Convert to time domain signal

# # Plot FID's
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(S):
#     Y = np.ones(n)*i
#     ax.plot(np.linspace(0,1,n),Y,np.abs(Data[i]), alpha = 1, zorder = -i)

# FData = np.fft.fft(Data)
# plt.figure()
# ax = plt.subplot(111)
# ax.plot(np.linspace(b2,b1,n),np.real(FData[0]))
# plt.title('Section of interest')
# ax.invert_xaxis()

t = np.linspace(0,1,n)

dwmin, dwmax, ddw = 35, 55, .01 # Domain of rates of change

Peaks = np.empty((0,4)) #array for peaks found

# #All series i one plot
# plt.figure()
# ax = plt.subplot(111)
# for i in range(S):
#     ax.plot(Scale,np.real(FData[i]), label='%f'%i)
# ax.invert_xaxis()

# exit()

#Functions to fit later
def Lorentz(x, A, V, B):
    
    return np.real(-1/(1j)*A*B/(V - x + 1j * B))
    # return np.real(A*B/(B - 1j*(V - x)))

def Lorentzcost(x0, dictionary):
    
    multiplycost = 1
    cost = dictionary['cut'] - Lorentz(dictionary['X'], dictionary['RA'], dictionary['V'], x0[0])
    # cost = dictionary['cut'] - Lorentz(dictionary['X'], x0[0], x0[1], x0[2])
    norm = np.linalg.norm(cost)
    if np.any(cost<0):
        multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
    return norm*multiplycost

def helper(V, A):
    
    def Lorentzhelp(x, B):
        return np.real(-1/(1j)*A*B/(V - x + 1j * B))
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
            p[i][k] = f[k]*np.e**(-b*k)
    
    # #Debugging
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(f)[0]),label='Pierwsza seria')
    # ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(f)[-1]),label='Ostatnia seria')
    # ax1.invert_xaxis()
    # ax1.legend()
    
    #3D plot of all series
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(s):
        Y = np.ones(n)*i
        ax.plot(np.arange(n),Y,np.real(np.fft.fft(f))[i], alpha = 1, zorder = -i)
    
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
    result = np.where(np.real(phat) == np.amax(np.real(phat))) # coordinates of global maxima
    # result = np.where(PR == np.amax(PR)) # coordinates of global maxima
    cords1, cords2 = result[0][0], result[1][0]
    dw1, w1 = DW[cords1], freq[cords2] # frequency and rate of change of highest peak
    
    return PR, cords1, cords2, w1, dw1, phat

def findbeta(Slice, V, RA):
    
    rinterval = 10
    cut = Slice[V-rinterval:V+rinterval]
    X = np.linspace(V-rinterval,V+rinterval-1,len(cut))
    
    #Initial parameters fit
    Tofit = helper(V, np.real(RA))
    popt,_ = curve_fit(Tofit, X, cut, bounds=((0), (15)))
    fit = Tofit(X, *popt)
    Bini = popt[0]
    
    dictionary = {'cut':cut,'X':X, 'RA':RA, 'V':V}
    x0 = [Bini] # inital B value
    
    #Optimizing optimization algorithms
    OptParams = minimize(Lorentzcost, x0, args=(dictionary), method='COBYLA')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(X,cut)
    ax.plot(X, fit, 'r-', label='initial fit')
    plt.vlines([V], 0, RA) # to check if alignes correctly
    ax.plot(X, Lorentz(X, RA, V, OptParams.x[0]), label='final fit')
    ax.legend()
    
    fig = plt.figure()
    plt.plot(np.arange(n),Slice, label='przekrój')
    plt.plot(np.arange(n),Lorentz(np.arange(n), RA, V, OptParams.x[0]), label='final fit')
    plt.legend()
    
    # plt.figure()
    # plt.plot(np.arange(n),Slice-Lorentz(np.arange(n), RA, V, OptParams.x[0]))
    
    # print("Beta found: %f" % OptParams.x[2])
    return np.abs(OptParams.x[0])

def findA(f, w, dw, B):
    
    A = 0
    C = 0
    for i in range(0,np.shape(f)[0]):
        for j in range(0,np.shape(f)[1]):
            A += np.conjugate(np.e**(
            (2*np.pi*1j*(w+i*dw+1j*B)*j/np.shape(f)[1])))*f[i][j]
            #np.e**((-2*np.pi*1j*(w+i*dw-1j*B)*j/np.shape(f)[1]))*f[i][j]
            C += np.conjugate(np.e**(
            (2*np.pi*1j*(w+i*dw+1j*B)*j/np.shape(f)[1])))*np.e**(
            (2*np.pi*1j*(w+i*dw+1j*B)*j/np.shape(f)[1]))
            #np.e**(2*np.pi*(-2*B*j/np.shape(f)[1]))
    return A/C

def subtractpeak(f, w, dw, B, A):
    
    ffit = np.zeros((np.shape(f)[0],np.shape(f)[1]),dtype="complex64")
    for i in range(np.shape(f)[0]):
        ffit[i] = np.add(ffit[i], A*np.e**((2*np.pi*1j*(w+i*dw+1j*B)*t)))
    
    for i in range(np.shape(ffit)[0]):
        ffit[i][0] /= 2
    
    Radon(ffit,dwmin,dwmax,ddw)
    fnext = f-ffit
    # subtracted, cords1, w1, dw1 = Radon(fnext,-2,2,.01)
    # print("Peak subtracted at:" + "\n" + "w = {:.2f}, dw = {:.2f}, B = {:.2f}, A = {:.2f}".format(w, dw, B, A))
    return fnext

def dothething(f,dwmin,dwmax,ddw, Peaks):
    
    PR, cords1, cords2, w, dw, phat = Radon(f,dwmin,dwmax,ddw)
    slicereal = np.real(phat[cords1])
    RA = (phat[cords1][cords2]).real
    # RA = np.max(np.real(phat[cords]))
    B = findbeta(slicereal, w, RA)    
    A = findA(f, w, dw, B)
    fnext = subtractpeak(f, w, dw, B, A)
    Peaks = np.append(Peaks, [[w,dw,B,A]], axis=0)

    # print(b2-w*abs(b2-b1)/n)
    # print(b2-(w-S*dw)*abs(b2-b1)/n)
    # print(abs(b2-b1)/n) # resolution
    return fnext, Peaks

# Main()

d = {}
d["Data0"] = Data
Progress = [' '] * iterations
for i in range(iterations):
    print("Progress: "+str(Progress)+" \r",)
    d["Data{0}".format(i+1)], Peaks = dothething(d["Data{0}".format(i)],dwmin,dwmax,ddw, Peaks)
    Progress[i] = '#'
print("Progress: "+str(Progress))

print('\nSummary:\n')
for i in range(len(Peaks)):
    print('\t'+'Peak {0}\n'.format(i+1)+
          'Frequency:\n'+str(np.real(Peaks[i][0]))+' ppm'+'\n'+
          'Speed:\n'+str(np.real(Peaks[i][1]))+' ppb/K'+'\n'+
          'Damping:\n'+str(np.real(Peaks[i][2]))+'\n'+
          'Amplitude:\n'+str(Peaks[i][3])+'\n')

print('It took', time.time()-start, 'seconds.')