#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:31:05 2018

@author: gcf
"""

from abm4py import core
import numpy as np
import math

#%% numba module is available
if core.NUMBA:

    @core.njit("f8 (f8[:], f8[:])",cache=True)
    def cobbDouglasUtilNumba(x, alpha):
        utility = 1.
        
        for i in range(len(x)):
            utility = utility * (100.*x[i])**alpha[i]    
        
        return utility / 100.
    
    @core.njit("f8[:] (f8[:])",cache=True)
    def normalize(array):
        return  array / np.sum(array)
    
    @core.njit(cache=True)
    def sum1D(array):
        return np.sum(array)
    
    @core.njit(cache=True)
    def sumSquared1D(array):
        return np.sum(array**2)
    
    @core.njit(cache=True)
    def prod1D(array1, array2):
        return np.multiply(array1,array2)
    
    @core.njit(cache=True)
    def normalizedGaussian(array, center, errStd):
        diff = (array - center) +  np.random.randn(array.shape[0])*errStd
        normDiff = np.exp(-(diff**2.) / (2.* errStd**2.))  
        return normDiff / np.sum(normDiff)
    
    @core.njit(cache=True)
    def convenienceFunction(minValue, maxValue, delta, mu, twoSigmaSquare, density):
        return  minValue + delta * math.exp(-(density - mu)**2 / twoSigmaSquare)
    
    @core.njit(cache=True)
    def convenienceFunctionArray(minValue, maxValue, delta, mu, twoSigmaSquare, density):
        return  minValue + delta * np.exp(-(density - mu)**2 / twoSigmaSquare)
    
else:
    def cobbDouglasUtilNumba(x, alpha):
        utility = 1.
        
        for i in range(len(x)):
            utility = utility * (100.*x[i])**alpha[i]    
        
        return utility / 100.
    

    def normalize(array):
        return  array / np.sum(array)
    
    def sum1D(array):
        return np.sum(array) 

    def sumSquared1D(array):
        return np.sum(array**2)
    
    def prod1D(array1, array2):
        return np.multiply(array1,array2)    

    def normalizedGaussian(array, center, errStd):
        diff = (array - center) +  np.random.randn(array.shape[0])*errStd
        normDiff = np.exp(-(diff**2.) / (2.* errStd**2.))  
        return normDiff / np.sum(normDiff)    

    def convenienceFunction(minValue, maxValue, delta, mu, twoSigmaSquare, density):
        return  minValue + delta * math.exp(-(density - mu)**2 / twoSigmaSquare)    

    def convenienceFunctionArray(minValue, maxValue, delta, mu, twoSigmaSquare, density):
        return  minValue + delta * np.exp(-(density - mu)**2 / twoSigmaSquare)