import numpy as np
import csv
import matplotlib.pyplot as plt
import mayavi.mlab as may
from scipy.signal import hilbert
import scipy.ndimage.filters as filters


class Params:
    
    def __init__(self,A,w,B,dw,S,N):
        
        self.A = A # amplitudes
        self.w = w # frequencies
        self.B = B # damping coeffs
        self.dw = dw # changes in frequencies
        self.S = S # number of series
        self.N = N # number of points in serie
    
    
class Radon:

    def __init__(self, Data, dwmin, dwmax, ddw, *args):
        
        '''Data can be a file location string
        a Params class object containing parameters
        for creating model peaks'''
        
        self.dwmin = dwmin
        self.dwmax = dwmax
        self.ddw = ddw
            
        if (type(Data) == str) or ((args[0] if len(args) > 0 else None) == 'file'):
            
            # Import data from file
            Dataspectrum = np.array(list(csv.reader(open(Data), delimiter="\t")))
            
            # Extract scale in ppm
            Scale = Dataspectrum[1:,0].astype('float64')
            self.b1, self.b2 = float(Scale[0]), float(Scale[-1]) # b1 - smallest, b2 - biggest
            
            # Create data matrix (M[number of series][number of spectrum points])
            Dataspectrum = np.delete(np.delete(np.delete(Dataspectrum,0,0),-1,1),0,1)
            Dataspectrum = Dataspectrum.astype('float64')
            Dataspectrum = Dataspectrum.transpose()
            
            # Matrix of series of spectra
            Spectra = hilbert(Dataspectrum,axis=1)
            self.Spectra = np.conj(Spectra)
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.FID = np.fft.ifft(self.Spectra,self.n,axis=1) # Convert to time domain signal
            self.t = np.linspace(0,1,self.n)
            
            self.Scale = np.linspace(self.b1,self.b2,self.n) # linearize scale
            
            self.abs_Radon = None
            self.complex_Radon = None
            self.frequency = None
            self.speed = None
            self.transform()
        
        elif isinstance(Data, Params) or ((args[0] if len(args) > 0 else None) == 'Params'):
            
            #Create FID of model peaks
            FID = np.zeros((Data.S,Data.N),dtype="complex_")
            t = np.linspace(0,1,Data.N,endpoint=False)
            for i in range(Data.S):
                for k in range(np.shape(Data.A)[0]):
                    FID[i] = np.add(FID[i], Data.A[k]*np.e**(
                        (2*np.pi*1j*(Data.w[k]+i*Data.dw[k]+1j*Data.B[k])*t)
                        ))
                    # f[i] = np.add(f[i],.5*np.random.randn(n)) # Add random noise
            
            #fix the first point for Fourier Transform
            for i in range(Data.S):
                FID[i][0] /= 2
            
            self.FID = FID
            self.Spectra = np.fft.fft(FID)
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.t = t
            
            self.abs_Radon = None
            self.complex_Radon = None
            self.frequency = None
            self.speed = None
            self.transform()
        
        elif (args[0] if len(args) > 0 else None) == 'Spectra':
            
            self.Spectra = Data
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.FID = np.fft.ifft(self.Spectra,self.n,axis=1) # Convert to time domain signal
            self.t = np.linspace(0,1,self.n,endpoint=False)
            
            self.abs_Radon = None
            self.complex_Radon = None
            self.frequency = None
            self.speed = None
            self.transform()
        
            
    #Radon Transform
    def transform(self):
        
        #Phase correction
        DW = np.arange(self.dwmin,self.dwmax,self.ddw) # domain of rates of change
        p = np.zeros((len(DW),self.s,self.n),dtype="complex64")
        a = 2*np.pi*1j*self.t
        for i in range(len(DW)):
            b = a*DW[i]
            for k in range(self.s):
                p[i][k] = self.FID[k]*np.e**(-b*k)
            
        # #Debugging
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(FID)[0]),label='Pierwsza seria')
        # ax1.plot(np.linspace(b2,b1,n),np.real(np.fft.fft(FID)[-1]),label='Ostatnia seria')
        # ax1.invert_xaxis()
        # ax1.legend()
        
        #"Diagonal" summation
        Pr = np.zeros((len(DW),self.n),dtype="complex64")
        for i in range(len(DW)):
            for j in range(self.s):
                Pr[i] += p[i][j]
        
        freq = np.arange(self.n)
        
        #Fourier Transform
        phat = np.zeros((len(DW),self.n),dtype="complex64")
        PR = np.zeros((len(DW),self.n))
        phat = np.fft.fft(Pr)
        PR = np.abs(phat)
        
        self.abs_Radon = PR
        self.complex_Radon = phat
        
        
        #Peak Picker
        
        neighborhood_size = 10
        data_max = filters.maximum_filter(self.complex_Radon.real,
                                          neighborhood_size)
        maxima = (self.complex_Radon.real == data_max)
        threshold = (self.complex_Radon.real >= 0.7*np.max(self.complex_Radon.real))
        results = np.where(np.logical_and(maxima,threshold))
        
        dws = DW[results[0]]
        ws = freq[results[1]]
        
        self.frequency = ws
        self.speed = dws
                

    def Plot_abs(self):
        
        # Plotting the absolute value of radon spectrum
        may.figure()
        X, Y = np.mgrid[0:np.shape(self.abs_Radon)[0],
                        0:np.shape(self.abs_Radon)[1]]
        may.surf(X, Y, self.abs_Radon,
                 warp_scale=np.min(np.shape(self.abs_Radon))/np.max(self.abs_Radon))

    def Plot_real(self):
        
        # Plotting the real value of radon spectrum
        may.figure()
        X, Y = np.mgrid[0:np.shape(self.complex_Radon.real)[0],
                        0:np.shape(self.complex_Radon.real)[1]]
        may.surf(X, Y, self.complex_Radon.real,
                 warp_scale=np.min(np.shape(self.complex_Radon.real))/np.max(self.complex_Radon.real))


# Additional functions

def Plot_first_and_last(R):

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(R.Scale,np.real(R.Spectra[0]),label='1')
    ax.plot(R.Scale,np.real(R.Spectra[-1]),label='-1')
    plt.title('Full spectrum')
    ax.legend()
    ax.invert_xaxis()


def Plot_FIDs(R):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(R.s):
        Y = np.ones(R.n)*i
        ax.plot(np.linspace(0,1,R.n),Y,np.abs(R.FID[i]), alpha = 1, zorder = -i)


def Plot_all_series_spectra(R):
    
    plt.figure()
    ax = plt.subplot(111)
    for i in range(R.s):
        ax.plot(R.Scale,np.real(R.Spectra[i]), label='%f'%i)
    ax.invert_xaxis()

#3D plot of all series
def Plot_spectra_3D(R):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(R.s):
        Y = np.ones(R.n)*i
        ax.plot(np.arange(R.n),Y,
                np.real(np.fft.fft(R.FID))[i],
                alpha = 1, zorder = -i)