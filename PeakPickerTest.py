import numpy as np
from scipy.optimize import minimize
from sys import exit
from RadonClass import Radon, Params, Plot_first_and_last, Plot_FIDs, Plot_all_series_spectra, Plot_spectra_3D

# iterations = int(input('How many iterations?\n'))

import time
start = time.time()

dwmin, dwmax, ddw = 0, 3, .01 # Domain of rates of change
Peaks = np.empty((0,4)) #array for peaks found


# Example Data
A = np.array([2,2,0]) # amplitudes
w = np.array([100.,110.,0]) # frequencies
B = np.array([2,2.5,0]) # damping coeffs
dw = np.array([1.5,1.6,0]) # changes in frequencies

params = Params(A,w,B,dw,20,256)

R1 = Radon(params, dwmin, dwmax, ddw) # Instance of Radon class

# Plot_first_and_last(R1)
# R1.Plot_abs()
R1.Plot_real()
# Plot_FIDs(R1)
# Plot_all_series_spectra(R1)
Plot_spectra_3D(R1)


#Function to fit peaks to spectra
def FitPeaks(R, Peaks, *args):
    '''N is the number of peaks to fit, args[0] are
    the frequencies and args[1] are the speeds'''

    N = np.shape(R.frequency)[0] # Number of relevant peaks
    Rcut = R
    # # Cut out the area of interest
    # w_lower_bound = max(min(R.frequency)-50,0)
    # w_upper_bound = min(max(R.frequency)+50,R.n)
    # SpectraCut = R.Spectra[:,w_lower_bound:w_upper_bound]
    # dw_lower_bound = max(min(R.speed)-1,R.dwmin)
    # dw_upper_bound = min(max(R.speed)+1,R.dwmax)
    # Rcut = Radon(SpectraCut, dw_lower_bound,
    #              dw_upper_bound, R.ddw, 'Spectra')
    # Rcut.Plot_real()
    # Plot_spectra_3D(Rcut)
    
    # Function for finding parameters
    def Fit(x0, Rcut):
        
        Afit = x0[0:N]
        wfit = Rcut.frequency
        Bfit = x0[N:2*N]
        dwfit = Rcut.speed
        FitParams = Params(Afit,wfit,Bfit,dwfit,Rcut.s,Rcut.n)
        # print(FitParams.A,
        #       FitParams.w,
        #       FitParams.dw,
        #       FitParams.B,
        #       FitParams.N,
        #       FitParams.S)
        cost = Rcut.complex_Radon.real - Radon(FitParams,Rcut.dwmin, Rcut.dwmax, R.ddw).complex_Radon.real
        norm = np.linalg.norm(cost)
        print(norm)
        multiplycost = 1
        # if np.any(cost<0):
        #     multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
        
        return norm*multiplycost

    x0 = tuple([1]*2*N)
    OptParams = minimize(Fit, x0, args=(Rcut), method='COBYLA',
                          options={'maxiter': None})

    print(OptParams.x)
    
    B = OptParams.x[N:]
    A = OptParams.x[:N]
    # # No fitting
    # B = [2,2.5]
    # A = [2,2]
    
    Radon(Params(A,Rcut.frequency,B,Rcut.speed,Rcut.s,Rcut.n),
          Rcut.dwmin, Rcut.dwmax, R.ddw).Plot_real()
    
    
    for i in range(N):
        Peaks = np.append(Peaks, [[R.frequency[i],R.speed[i],B[i],A[i]]], axis=0)

    return Peaks


Peaks = FitPeaks(R1, Peaks)

# Main()

print('\nSummary:\n')
for i in range(len(Peaks)):
    print('\t'+'Peak {0}\n'.format(i+1)+
          'Frequency:\n'+str(round(np.real(Peaks[i][0]),4))+' ppm'+'\n'+
          'Speed:\n'+str(round(np.real(Peaks[i][1]),4))+' ppb/K'+'\n'+
          'Damping:\n'+str(round(np.real(Peaks[i][2]),4))+'\n'+
          'Amplitude:\n'+str(round(Peaks[i][3],4))+'\n')

print('It took', time.time()-start, 'seconds.')