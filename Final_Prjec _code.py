from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pylab
import matplotlib.pyplot as plt
import pandas as pd 


# getting the data

SP500 = pdr.get_data_yahoo('^GSPC', start="1992-04-01", end="2018-04-01")
FTSE = pdr.get_data_yahoo('^FTSE', start="1992-04-01", end="2018-04-01")
HSI = pdr.get_data_yahoo('HSI', start="1992-04-01", end="2018-04-01")
NASDAQ = pdr.get_data_yahoo('^IXIC', start="1992-04-01", end="2018-04-01")

SP500 = SP500['Adj Close']
FTSE = FTSE['Adj Close']
HSI = HSI['Adj Close']
NASDAQ = NASDAQ['Adj Close']

def Stats_and_Normality_tests(sym):
    
    array = np.array(sym)
    log_ret = np.log(array[1:] / array[0:-1])
    
    print('\n  ----------- Statistics -----------  \n')
    print('size',np.size(log_ret))
    print('min',np.min(log_ret))
    print('max',np.max(log_ret))
    print('mean',np.mean(log_ret))
    print('std',np.std(log_ret))
    
    print('\n  ----------- Normality test -----------  \n')
    print('skew of data set',stats.skew(log_ret))
    print('skew test p-value',stats.skewtest(log_ret)[1])
    print('kurtosis of data set',stats.kurtosis(log_ret))
    print('kurtosis test p-value',stats.kurtosistest(log_ret)[1])
    print('Normal test p-value',stats.normaltest(log_ret)[1])


print('\n  ----------- Test for a Log-normal Distribution of Prices -----------  \n')

print('\n  ----------- SP500 -----------  \n')
Stats_and_Normality_tests(SP500)

print('\n  ----------- FTSE -----------  \n')
Stats_and_Normality_tests(FTSE)

print('\n  ----------- HSI -----------  \n')
Stats_and_Normality_tests(HSI)

print('\n  ----------- NASDAQ -----------  \n')
Stats_and_Normality_tests(NASDAQ)


# Q-Q plots for testing log-normal distribution

pylab.rcParams['figure.figsize'] = (12, 10)

plt.subplot(2, 2, 1)
stats.probplot(np.log(np.array(SP500.values[1:]/SP500.values[0:-1])), dist="norm", plot=pylab)
plt.grid(True)
plt.title('Q-Q plot for SP500')
pylab.tight_layout()


plt.subplot(2, 2, 2)
stats.probplot(np.log(np.array(FTSE.values[1:]/FTSE.values[0:-1])), dist="norm", plot=pylab)
plt.grid(True) 
plt.title('Q-Q plot for FTSE')
pylab.tight_layout()

plt.subplot(2, 2, 3)
stats.probplot(np.log(np.array(HSI.values[1:]/HSI.values[0:-1])), dist="norm", plot=pylab)
plt.grid(True) 
plt.title('Q-Q plot for HSI')
pylab.tight_layout()

plt.subplot(2, 2, 4)
stats.probplot(np.log(np.array(NASDAQ.values[1:]/NASDAQ.values[0:-1])), dist="norm", plot=pylab)
plt.grid(True) 
plt.title('Q-Q plot for NASDAQ')
pylab.tight_layout()

plt.show()


# Normal Distribution of the log data

plt.subplot(2, 2, 1)
ss = np.array(SP500)
a = np.log(ss[1:] / ss[0:-1])
a1 = np.sort(a)
fit = stats.norm.pdf(a1, np.mean(a), np.std(a))  #this is a fitting indeed
plt.plot(a1, fit,'-o')
plt.hist(a1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('SP500 Normal ditribution of Log data')
plt.grid(True) 



plt.subplot(2, 2, 2)
ss = np.array(FTSE)
b = np.log(ss[1:] / ss[0:-1])
b1 = np.sort(b)
fit = stats.norm.pdf(b1, np.mean(b), np.std(b))  #this is a fitting indeed
plt.plot(b1, fit,'-o')
plt.hist(b1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('FTSE Normal ditribution of Log data')
plt.grid(True) 
 

plt.subplot(2, 2, 3)
ss = np.array(HSI)
c = np.log(ss[1:] / ss[0:-1])
c1 = np.sort(c)
fit = stats.norm.pdf(c1, np.mean(c), np.std(c))  #this is a fitting indeed
plt.plot(c1, fit,'-o')
plt.hist(c1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('HSI Normal ditribution of Log data')
plt.grid(True) 


plt.subplot(2, 2, 4)
ss = np.array(NASDAQ)
d = np.log(ss[1:] / ss[0:-1])
d1 = np.sort(d)
fit = stats.norm.pdf(d1, np.mean(d), np.std(d))  #this is a fitting indeed
plt.plot(d1, fit,'-o')
plt.hist(d1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('NASDAQ Normal ditribution of Log data')
plt.grid(True) 


plt.show()



print('\n  ----------- Test for a Normal Distribution of Returns -----------  \n')

Ret_SP500 = (SP500[1:].values - SP500[:-1].values) / SP500[:-1].values
print('Normal SP500 returns',stats.normaltest(Ret_SP500))

Ret_FTSE = (FTSE[1:].values - FTSE[:-1].values) / FTSE[:-1].values
print('Normal FTSE returns',stats.normaltest(Ret_FTSE))

Ret_HSI = (HSI[1:].values - HSI[:-1].values) / HSI[:-1].values
print('Normal HSI returns',stats.normaltest(Ret_HSI))

Ret_NASDAQ = (NASDAQ[1:].values - NASDAQ[:-1].values) / NASDAQ[:-1].values
print('Normal NASDAQ returns',stats.normaltest(Ret_NASDAQ))


# Normal Distribution Test for the Indices returns 

plt.subplot(2, 2, 1)
a = (SP500[1:].values - SP500[:-1].values) / SP500[:-1].values
a1 = np.sort(a)
fit = stats.norm.pdf(a1, np.mean(a), np.std(a))  #this is a fitting indeed
plt.plot(a1, fit,'-o')
plt.hist(a1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('SP500 Normal ditribution of Returns')
plt.grid(True) 



plt.subplot(2, 2, 2)
b = (FTSE[1:].values - FTSE[:-1].values) / FTSE[:-1].values
b1 = np.sort(b)
fit = stats.norm.pdf(b1, np.mean(b), np.std(b))  #this is a fitting indeed
plt.plot(b1, fit,'-o')
plt.hist(b1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('FTSE Normal ditribution of Returns')
plt.grid(True) 
 

plt.subplot(2, 2, 3)
c = (HSI[1:].values - HSI[:-1].values) / HSI[:-1].values
c1 = np.sort(c)
fit = stats.norm.pdf(c1, np.mean(c), np.std(c))  #this is a fitting indeed
plt.plot(c1, fit,'-o')
plt.hist(c1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('HSI Normal ditribution of Returns')
plt.grid(True) 


plt.subplot(2, 2, 4)
d = (NASDAQ[1:].values - NASDAQ[:-1].values) / NASDAQ[:-1].values
d1 = np.sort(d)
fit = stats.norm.pdf(d1, np.mean(d), np.std(d))  #this is a fitting indeed
plt.plot(d1, fit,'-o')
plt.hist(d1, bins = 70,density=True)      #use this to draw histogram of your data
plt.title('NASDAQ Normal ditribution of Returns')
plt.grid(True) 


plt.show()

# Geometric Brownina Motion Simulation

'''
N_sim: number of simulations
T: horizon
dt: step length in years
sigma: volatility per year
mu: drift terms (moving average or long-term mean for stock returns)
S0: initial stock price
'''

from numpy.random import standard_normal
from numpy import array, zeros, sqrt, shape
from pylab import *

S0 = 100

T =1
dt =0.05
sigma = 0.2
mu = 1
N_Sim = 100

Steps=round(T/dt); #Steps in years
S = zeros([N_Sim, Steps], dtype=float)
x = range(0, int(Steps), 1)

for j in range(0, N_Sim, 1):
        S[j,0]= S0
        for i in x[:-1]:
                S[j,i+1]=S[j,i]+S[j,i]*(mu-0.5*pow(sigma,2))*dt+sigma*S[j,i]*sqrt(dt)*standard_normal();
        plot(x, S[j])

title('Simulations %d Steps %d Sigma %.6f Mu %.6f S0 %.6f' % (int(N_Sim), int(Steps), sigma, mu, S0))
xlabel('steps')
ylabel('stock price')
show()



# Value at risk calculation for the probability of crach


S0 = 282
r = 0.05
sigma = 0.02
T = 58/365
I = 1000000

ST = S0 * np.exp((r - 0.5 * sigma ** 2)* T + sigma * np.sqrt(T) * np.random.standard_normal(I))

R_gbm = np.sort(ST - S0)

percs = [0.0002, 0.01, 0.1, 2.5, 5.0, 10.0]
var = stats.scoreatpercentile(R_gbm,percs)

print ('Value at risk for SP500')
print ('%16s %16s' % ('Confidence Level', 'Value at Risk'))
print(35*'-')
for pair in zip(percs, var):
    print ('%16.4f %16.2f' % (100 - pair[0], -pair[1]))




S0 = 406
r = 0.05
sigma = 0.02
T = 46/365
I = 1000000

ST = S0 * np.exp((r - 0.5 * sigma ** 2)* T + sigma * np.sqrt(T) * np.random.standard_normal(I))

R_gbm = np.sort(ST - S0)

percs = [0.0002, 0.01, 0.1, 2.5, 5.0, 10.0]
var = stats.scoreatpercentile(R_gbm,percs)

print ('Value at risk for NASDAQ')
print ('%16s %16s' % ('Confidence Level', 'Value at Risk'))
print(35*'-')
for pair in zip(percs, var):
    print ('%16.4f %16.2f' % (100 - pair[0], -pair[1]))


S0 = 2301
r = 0.05
sigma = 0.02
T = 249/365
I = 1000000

ST = S0 * np.exp((r - 0.5 * sigma ** 2)* T + sigma * np.sqrt(T) * np.random.standard_normal(I))

R_gbm = np.sort(ST - S0)

percs = [0.0002, 0.01, 0.1, 2.5, 5.0, 10.0]
var = stats.scoreatpercentile(R_gbm,percs)

print ('Value at risk for FTSE')
print ('%16s %16s' % ('Confidence Level', 'Value at Risk'))
print(35*'-')
for pair in zip(percs, var):
    print ('%16.4f %16.2f' % (100 - pair[0], -pair[1]))


S0 = 3783
r = 0.05
sigma = 0.02
T = 420/365
I = 1000000

ST = S0 * np.exp((r - 0.5 * sigma ** 2)* T + sigma * np.sqrt(T) * np.random.standard_normal(I))

R_gbm = np.sort(ST - S0)

percs = [0.0002, 0.01, 0.1, 2.5, 5.0, 10.0]
var = stats.scoreatpercentile(R_gbm,percs)

print ('Value at risk for HSI')
print ('%16s %16s' % ('Confidence Level', 'Value at Risk'))
print(35*'-')
for pair in zip(percs, var):
    print ('%16.4f %16.2f' % (100 - pair[0], -pair[1]))



def Fat_tail_test(index):
    ret = (index[1:].values - index[:-1].values) / index[:-1].values
    print( 'size =',len(ret)) 
    print( 'mean =',round(np.mean(ret),8)) 
    print( 'std =',round(np.std(ret),8)) 
    print( 'skewness=',round(stats.skew(ret),8)) 
    print( 'kurtosis=',round(stats.kurtosis(ret),8))


print('\n  ----------- Fat-tail Test for SP500 returns -----------  \n')
Fat_tail_test(SP500)
print('\n  ----------- Fat-tail Test for FTSE returns -----------  \n') 
Fat_tail_test(FTSE)
print('\n  ----------- Fat-tail Test for HSI returns -----------  \n') 
Fat_tail_test(HSI)
print('\n  ----------- Fat-tail Test for NASDAQ returns -----------  \n') 
Fat_tail_test(NASDAQ)



NASDAQ2 = pdr.get_data_yahoo('^IXIC', start="2008-04-01", end="2018-04-01")

NAS = NASDAQ2['Adj Close']
Ret_NASDAQ = (NAS[1:].values - NAS[:-1].values) / NAS[:-1].values

ret1=Ret_NASDAQ
plt.plot(ret1)
plt.xlabel("Period")
plt.ylabel("Return")
plt.title("NASDAQ Return Evolution")
plt.show()

def hurst(n): #implement a function that returns the logarithm of the rescaled range
    retn=ret1[0:int(n)] #Microsoft stock return for n period
    yn=retn-np.mean(retn) #calculation of mean adjusted series for this period
    zn=np.cumsum(yn) # calculation of the cumulative sum
    Rn=np.max(zn)-min(zn) #calculation of range
    Sn=np.std(retn) # calculation of the standard deviation
    En=Rn/Sn # calculation of the rescaled range
    return np.log(En)

y=[hurst(np.size(ret1)), hurst(np.size(ret1)/2), hurst(np.size(ret1)/4),
hurst(np.size(ret1)/8), hurst(np.size(ret1)/16),hurst(np.size(ret1)/32)]

x=[np.log(np.size(ret1)), np.log(np.size(ret1)/2), np.log(np.size(ret1)/4), 
np.log(np.size(ret1)/8), np.log(np.size(ret1)/16), np.log(np.size(ret1)/32)]

x=sm.add_constant(x)
model=sm.OLS(y,x)
results=model.fit()
print (results.summary())




from __future__ import division 
import matplotlib.pyplot as plt 




# set the number of fractal iterations 
n = 3

# an array of color options for the different lines (letter corresponds to color, - corresponds to line type)
colors = ['g-', 'r-', 'y-', 'm-', 'c-', 'k-', 'w-']

# Initial x-coordinates -- these need to be in increasing order to plot correctly
initx = [0, 1, 2, 3, 4, 5]
# Initial y-coordinates
inity = [0, 3, 2, 4, 3, 5]


sortedY = sorted(inity)

# The domain should be the distance between the largest and smallest x
initDomain = initx[-1] - initx[0]
# The range should be the distance between the largest and smallest y
initRange = sortedY[-1] - sortedY[0]




if len(initx) != len(inity):
	print ("The two coordinate arrays are of different size")
	exit(0)


plt.plot(initx, inity, 'b-')


# Iterate the fractal by the number n, specified earlier
for j in range(n):

	# Initialize arrays to hold the x and y coordinates of the current iteration
	newx = []
	newy = []

	# Go iterate through every element of the initial array
	for i in range(len(initx)-1):
		# Get the domain between the current element and the next element
		currentDomain = initx[i+1] - initx[i]
		# Get the range between the current element and the next element
		currentRange = inity[i+1] - inity[i]

		# Since we want a scalar multiplier to apply to our points, make it
		# the ratio between the domain/range just found and the domain/range
		# of our initial graph (since that is essentially our fractal template)
		domainScale = currentDomain / initDomain
		rangeScale = currentRange / initRange

		# Apply the scaling factor to each x coordinate in the initial array in order 
		# to create a new set of points between the current and next element of the initial array
		for val in initx:
			# Scale, and add the amount of the current element in the initial array so it
			# is offset to the proper location
			x = (val*domainScale) + initx[i]
			newx.append(x)

		# Apply the scaling factor to each y coordinate in the initial array in order 
		# to create a new set of points between the current and next element of the initial array
		for val in inity:
			# Scale, and add the amount of the current element in the initial array so it
			# is offset to the proper location
			y = (val*rangeScale) + inity[i]
			newy.append(y)

	# Set the init arrays to be equal to the arrays we just built, so on the next pass
	# though the loop, we are advancing the level of the fractal.
	initx = newx
	inity = newy

	# Plot the current line
	plt.plot(newx, newy, colors[j])



# Label the y-axis
plt.ylabel('output')
# Label the x-axis
plt.xlabel('input')
# display the graph
plt.show()
