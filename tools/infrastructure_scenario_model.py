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
for year in range(2005,2018):
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'
    #fileName = '../resources_ger/charge_stations_' +str(year) + '_186x219.npy'
    chargeMap = np.load(fileName)
    xData.append(year)
    yData.append(np.nansum(chargeMap))
    dyData.append(np.nansum(chargeMap) - np.sum(dyData))

xData.append(2020)
yData.append(70000) 

elStationsPara = dict()



#xx = pickle.load(open('chargingStationData.pkl', 'rb') )
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
    elStationsPara[maxStations] = popt

import pickle
fid = open('../resources/chargingStationPara.pkl', 'wb') 
pickle.dump(elStationsPara, fid)
fid.close()   

#%% 
if False:
    # correction of incomplete data
    maxStations =1e5
    def sigmoid(x, x0, k):
        y = (1. / (1 + np.exp(-k*(x-x0)))) * maxStations
        return y
 
    popt, pcov = curve_fit(sigmoid, xData, yData,p0=[2025, .5])
    print(popt)
    xProjection = np.linspace(2005, 2035, 31)
    yProjection = sigmoid(xProjection, *popt).astype(int)
    #%%
    iEnd = 10
    fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(xData[iEnd]) + '_186x219.npy'
    chargeMap = np.load(fileName)
    
    xList, yList = np.where(chargeMap>0)
    statX, statY = [], []
    for x,y in zip(xList.tolist(), yList.tolist()):
        print(x,y )
        statX.extend([x]*int(chargeMap[int(x),int(y)]))
        statY.extend([y]*int(chargeMap[int(x),int(y)]))
    
    rand = np.random.rand(len(statX))    
    randIdx = np.argsort(rand)
    #random ordered station
    statXRand = np.asarray(statX)[randIdx]
    
    statYRand = np.asarray(statY)[randIdx]
    
    plt.plot(xData[:iEnd], yProjection[:iEnd])
    
    # regernate map
    newMap = chargeMap*0
    plt.figure()
    plt.clf()
    expectedStattionList = []
    for ii in range(iEnd):
        nStations = int(yProjection[ii])
        expectedStattionList.append(nStations)
        newStat = np.asarray([statXRand[:nStations], statYRand[:nStations]])
        uniquePos, count = np.unique(newStat, axis=1, return_counts=True)
        #uniqueRandIdx, count = np.unique(randIdx,return_counts=True)
            
        newMap[uniquePos[0,:], uniquePos[1,:]] = count   
        print(xData[ii], np.nansum(newMap), expectedStattionList[ii])
        if ii > 1:
            
            plt.subplot(3,3,ii-1)
            plt.imshow(newMap)
            plt.clim([0, np.nanpercentile(newMap,98)])
            plt.colorbar()
        fileName = 'charge_stations_' + str(xData[ii]) + '_186x219.npy'
        np.save(fileName, newMap)
        
    # add okayisch data
    for year in range(2015, 2019):
        fileName = '/home/gcf/shared/processed_data/tiff/e_charging_stations_ger/charge_stations_' + str(year) + '_186x219.npy'    
        chargeMap = np.load(fileName)
        if year == 2015:
            plt.subplot(3,3,9)
            plt.imshow(chargeMap)
            plt.clim([0, np.nanpercentile(chargeMap,98)])
            plt.colorbar()
        print(year, np.nansum(chargeMap))
        fileName = 'charge_stations_' + str(year) + '_186x219.npy'    
        np.save(fileName, chargeMap)
    plt.tight_layout()