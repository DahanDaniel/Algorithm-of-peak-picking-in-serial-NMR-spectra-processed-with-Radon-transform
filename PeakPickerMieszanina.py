import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sys import exit
from RadonClass import *
import copy

iterations = int(input('How many iterations?\n'))

import time
start = time.time()

# Import data # Mieszanina
file = r"C:\Daniel\Programowanie\Radon_transform\sample_data\Osocze_wspolczynniki_temperaturowe\Mieszanina.csv"

dwmin, dwmax, ddw = 35, 55, .05 # Domain of rates of change
Peaks = np.empty((0,4)) #array for peaks found

R1 = Radon(file, dwmin, dwmax, ddw) # Instance of Radon class

Plot_first_and_last(R1,1)
# R1.Plot_abs()
# R1.Plot_real()

# R1.trim(3,3.15)
R1.trim(3,4.2)

Plot_first_and_last(R1,1)
# Plot_FIDs(R1)
# Plot_all_series_spectra(R1)
# Plot_spectra_3D(R1)

R1.Plot_real()

# exit()

#Function to fit peaks to spectra
def FitPeaks(R, Peaks, *args):
    
    '''N is the number of peaks to fit, args[0] are
    the frequencies and args[1] are the speeds'''

    # # Cut out the area of interest
    radius = 256
    lower_bound, upper_bound = max(0,R.frequency[0]-radius), min(R.n,R.frequency[0]+radius)
    # Check if boudaries are within spectra edges
    if lower_bound == 0: upper_bound = 2*radius
    elif upper_bound == R.n: lower_bound = R.n - 2*radius
    R.trim(toppm(R,lower_bound),toppm(R,upper_bound))
    R.Plot_real()
    Plot_first_and_last(R,1)
    
    N = np.shape(R.frequency)[0] # Number of relevant peaks
    
    # Function for finding parameters
    def Fit(x0, R): 
        
        Afit = x0[0:N]
        wfit = R.frequency
        Bfit = x0[N:2*N]
        dwfit = R.speed
        FitParams = Params(Afit,wfit,Bfit,dwfit,R.s,R.n)
        # print(FitParams.A,
        #       FitParams.w,
        #       FitParams.dw,
        #       FitParams.B,
        #       FitParams.N,
        #       FitParams.S)
        cost = R.complex_Radon.real - Radon(FitParams,R.dwmin, R.dwmax, R.ddw).complex_Radon.real
        norm = np.linalg.norm(cost)
        print(norm)
        multiplycost = 1
        # if np.any(cost<0):
        #     multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
        
        return norm*multiplycost

    x0 = tuple([1]*2*N)
    OptParams = minimize(Fit, x0, args=(R), method='Nelder-Mead', # 'COBYLA' or 'Nelder-Mead'
                          options={'maxiter': None})

    # print(OptParams.x)
    
    B = OptParams.x[N:]
    A = OptParams.x[:N]
    # # No fitting
    # B = [2,2.5]
    # A = [2,2]
    
    # Fited object
    Rfit = Radon(Params(A,R.frequency,B,R.speed,R.s,R.n),
                 R.dwmin, R.dwmax, R.ddw, b1=R.b1, b2=R.b2)
    
    # Plot fitted Radon spectrum
    Rfit.Plot_real()

    # # Plot residuals
    # print(R.complex_Radon.real - Rfit.complex_Radon.real)
    
    for i in range(N):
        Peaks = np.append(Peaks, [[lower_bound+R.frequency[i],R.speed[i],B[i],A[i]]], axis=0)
        # Peaks = np.append(Peaks, [[toppm(R,R.frequency[i]),R.speed[i],B[i],A[i]]], axis=0)

    return Rfit, Peaks


def dothething(R, Peaks):
    
    Rcopy = copy.deepcopy(R)
    Rfit, Peaks = FitPeaks(Rcopy, Peaks)
    R1.subtract(Rfit)
    R1.Plot_real()
    
    # print(np.max(R1.complex_Radon.real))
    return R1, Peaks


# Main()

for i in range(iterations):
    R1, Peaks = dothething(R1, Peaks)
    
# Output file
with open("peaks.txt","a+") as output:
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    output.write(dt_string + '\n')

    output.write('\nSummary:\n')
    for i in range(len(Peaks)):
        output.write('\t'+'Peak {0}\n'.format(i+1)+
          'Shift:\n'+str(round(toppm(R1,int(np.real(Peaks[i][0]))),5))+' ppm'+'\n'+
          'Change:\n'+str(round(np.real(Peaks[i][1])*1000*abs(R1.b1-R1.b2)/R1.n,3))+' ppb/K'+'\n'+
          'Damping:\n'+str(round(np.real(Peaks[i][2]),4))+'\n'+
          'Amplitude:\n'+str(round(Peaks[i][3],4))+'\n')

    output.write('\n'+'It took ' + str(time.time()-start) + ' seconds.' + '\n')

# print('\nSummary:\n')
# for i in range(len(Peaks)):
#     print('\t'+'Peak {0}\n'.format(i+1)+
#           'Frequency:\n'+str(round(toppm(R1,int(np.real(Peaks[i][0]))),5))+' ppm'+'\n'+
#           'Speed:\n'+str(round(np.real(Peaks[i][1])*1000*abs(R1.b1-R1.b2)/R1.n,3))+' ppb/K'+'\n'+
#           'Damping:\n'+str(round(np.real(Peaks[i][2]),4))+'\n'+
#           'Amplitude:\n'+str(round(Peaks[i][3],4))+'\n')

# print('It took', time.time()-start, 'seconds.')