#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:32:31 2018

@author: gcf
"""

import numpy as np
#import mod_geotiff as gt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pylab

xData   = list()
yData   = list()
dyData  = [0]
for year in range(2005,2019):
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'
    chargeMap = np.load(fileName)
    xData.append(year)
    yData.append(np.nansum(chargeMap))
    dyData.append(np.nansum(chargeMap) - np.sum(dyData))

xData.append(2020)
yData.append(70000) 

plt.figure('fit', figsize=(6,3))
plt.clf()
pylab.plot(xData, yData, 'o', label='Daten')
pylab.plot(2020,70000, 'd', label='Regierungsziel')
for maxStations in [2e5, 5e5, 1e6]:
    def sigmoid(x, x0, k):
        y = (1. / (1 + np.exp(-k*(x-x0)))) * maxStations
        return y
 
    popt, pcov = curve_fit(sigmoid, xData, yData,p0=[2025, .5])
    print(popt)
    
    xProjection = np.linspace(2005, 2035, 31)
    yProjection = sigmoid(xProjection, *popt)
    pylab.plot(xProjection, yProjection, '--', label='max Stations = ' + str(int(maxStations)))
    