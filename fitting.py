# -*- coding: utf-8 -*-

filename="phylab2mydata.txt"
# change this if your filename is different


import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt


data=loadtxt(filename, usecols=(0,1,2,3), skiprows=1, unpack=True)
# load filename, take columns 0 & 1 & 2 & 3, skip 1 row, unpack=transpose x&y

xdata=data[0]
ydata=data[1]
xerror=data[2]
yerror=data[3]
# finished importing data, naming it sensibly

def my_func(phi,t0,B,C,D,E,F):
    return t0 + B * phi + C * phi ** 2 + D * phi ** 3 + E * phi ** 4 + F * phi ** 5
# this is the function we want to fit. the first variable must be the
# x-data (time), the rest are the unknown constants we want to determine

init_guess=(0,0,0,0,0,0)
# your initial guess of (a,tau,T,phi)

popt, pcov = optimize.curve_fit(my_func, xdata, ydata, p0=init_guess, maxfev=1000000)
# we have the best fit values in popt[], while pcov[] tells us the uncertainties

t0=popt[0]
B=popt[1]
C=popt[2]
D=popt[3]
E=popt[4]
F=popt[5]
# best fit values are named nicely
u_t0=pcov[0,0]**(0.5)
u_B=pcov[1,1]**(0.5)
u_C=pcov[2,2]**(0.5)
u_D=pcov[3,3]**(0.5)
u_E=pcov[4,4]**(0.5)
u_F=pcov[5,5]**(0.5)
# uncertainties of fit are named nicely

def fitfunction(phi):
    return t0 + B * phi + C * phi ** 2 + D * phi ** 3 + E * phi ** 4 + F * phi ** 5
#fitfunction(t) gives you your ideal fitted function, i.e. the line of best fit


start=min(xdata)
stop=max(xdata)    
xs=np.arange(start,stop,(stop-start)/1000) # fit line has 1000 points
curve=fitfunction(xs)
# (xs,curve) is the line of best fit for the data in (xdata,ydata) 

fig, (ax1,ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.6)
#hspace is horizontal space between the graphs

ax1.errorbar(xdata,ydata,yerr=yerror,xerr=xerror,fmt=".")
# plot the data, fmt makes it data points not a line

ax1.plot(xs,curve)
# plot the best fit curve on top of the data points as a line

ax1.set_xlabel("Amplitude")
ax1.set_ylabel("Peroid")
ax1.set_title("Best fit of some data points")
# HERE is where you change how your graph is labelled


print("T0:", t0, "+/-", u_t0)
print("B:", B, "+/-", u_B)
print("C:", C, "+/-", u_C)
print("D:", D, "+/-", u_D)
print("E:", E, "+/-", u_E)
print("F:", F, "+/-", u_F)
# prints the various values with uncertainties

residual=ydata-fitfunction(xdata)
# find the residuals
zeroliney=[0,0]
zerolinex=[start,stop]
# create the line y=0

ax2.errorbar(xdata,residual,yerr=yerror,xerr=xerror,fmt=".")
# plot the residuals with error bars

ax2.plot(zerolinex,zeroliney)
# plotnthe y=0 line on top

ax2.set_xlabel("Amplitude")
ax2.set_ylabel("Residuals of the amplitude")
ax2.set_title("Residuals of the fit")
# HERE is where you change how your graph is labelled

plt.show()
# show the graph