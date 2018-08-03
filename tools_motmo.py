#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

---- MoTMo ----
MOBILITY TRANSFORMATION MODEL
-- INIT FILE --

This file is part on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.


"""
import numpy as np
import time
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import classes_motmo as motmo
from gcfabm import core
doPlot = False

en = core.enum

def assess_km_share_age(earth, ):
    """
    Method to assess the share of km per mobility type classified bye person age
    """
    enums = earth.getEnums() 
    
    mobTypeIDs = [0, 2, 4]
    if not hasattr(earth, 'calShareAge'):
        #read  he calibration data and save it to the world
        dfShareAge = pd.read_csv( 'resources/calData_age_km_share_2008.csv', index_col=[0,1], header=0)
        earth.calShareAge = dfShareAge
    
    # This dictionary filters the correct age classes and is given to the related earth
    # method 
    ageFilterDict = OrderedDict([
            (5 , lambda a : (a['age']>=18) & (a['age'] < 25)),
            (6 , lambda a : (a['age']>=25) & (a['age'] < 45)),
            (7 , lambda a : (a['age']>=45) & (a['age'] < 60)),
            (8 , lambda a : (a['age']>=60) & (a['age'] < 65)),
            (9 , lambda a : (a['age']>=65))])
    # vector of assumed mean distances per trip
    # trips categories are 
    # 0-500m, 500m - 2.5km, 2.5-10km, 10-50km and < 50km
    MEAN_KM_PER_TRIP = [.25, 3., 7.5, 30., 75. ]
    
    
    
    #dataframe with the error measure
    errorDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])
    #dataframe with the corresponging values of the simulation
    simulationDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])
    # weighting array of the age categories (propotional to number of agents)
    weightOfCat = np.zeros(len(ageFilterDict))
    
    # loop over the age filers
    for i, cat in enumerate(ageFilterDict.keys()):
        #array of mobility choices of the group
        mobChoice = earth.getAttrOfFilteredAgentType('mobType', ageFilterDict[cat] ,3)
        
        weightOfCat[i] = len(mobChoice) 
        
        # estimated km per agent of the group        
        mobKmOfCat = np.dot(earth.getAttrOfFilteredAgentType('nJourneys', ageFilterDict[cat], 3),MEAN_KM_PER_TRIP)
        
        #loop over the 5 mobility choices which sums up hte km per tpye
        xx = [np.sum(mobKmOfCat[mobChoice==mobOpt]) for mobOpt in mobTypeIDs]
        
        # store the km share in the simulation array
        simulationDf.loc[cat] = xx / np.sum(xx)
        
        # store the absolute error in the error data frame
        errorDf.loc[cat] = np.abs(earth.calShareAge.iloc[i,:] - simulationDf.iloc[i,:])
        
    # normalize weights
    weightOfCat = weightOfCat / np.sum(weightOfCat)
    
    return errorDf, earth.calShareAge, simulationDf, weightOfCat


def assess_km_share_hh_type(earth, ):
    """
    Method to assess the share of km per mobility type classified bye household type
    """
    # hhtype from the simulation
    catDict = earth.getEnums()['hhTypes']
    mobDict = earth.getEnums()['mobilityTypes'].copy()
    # mobTypeIDs with available data
    mobTypeIDs = [0, 2, 4]
    
    if not hasattr(earth, 'calShareHHTye'):
        #read  he calibration data and save it to the world
        dfcalShareHHTye = pd.read_csv( 'resources/calData_household type_sum_kms_2008.csv', index_col=1, header=1)
        earth.calShareHHTye = dfcalShareHHTye.drop('hhtyp',axis=1).values
    
    # vector of assumed mean distances per trip
    # trips categories are 
    # 0-500m, 500m - 2.5km, 2.5-10km, 10-50km and < 50km
    MEAN_KM_PER_TRIP = [.25, 3., 7.5, 30., 75. ]
    
    #dataframe with the error measure
    errorDf     = np.zeros([len(catDict),len(mobTypeIDs)])
    #dataframe with the corresponging values of the simulation
    simulationDf= np.zeros_like(errorDf)

    # weighting array of the age categories (propotional to number of agents)
    weightOfCat = np.zeros(len(catDict))
    
    #get household ID of all persons
    hhIDs     = earth.getAttrOfAgentType('hhID',en.PERS)
    # get the household type of these households
    hhTypes   = earth.getAttrOfAgents('hhType', hhIDs)
    # get the mobility choice of all persons
    mobChoice = earth.getAttrOfAgentType('mobType' ,3)
    # get the number of journeys of all persons
    nJourneys = earth.getAttrOfAgentType('nJourneys',  3)

    # loop over all household types
    for i,cat in enumerate(catDict.keys()):
        # estimated km per agent of the group     
        mobKmOfCat     = np.dot(nJourneys[hhTypes==cat],MEAN_KM_PER_TRIP)
        #array of mobility choices of the group
        mobChoiceOfCat = mobChoice[hhTypes==cat]
        # weight proportial to the number of persons
        weightOfCat[i] = len(mobChoiceOfCat) 
        
        #loop over the 5 mobility choices which sums up hte km per tpye
        xx = [np.sum(mobKmOfCat[mobChoiceOfCat==mobOpt]) for mobOpt in mobTypeIDs]
        #store km share in the simulation data frame
        simulationDf[i,:] = xx / np.sum(xx)        
        #store the absulte difference in the error data frame
        errorDf[i,:] = np.abs(earth.calShareHHTye[i,[0,2,4]] - simulationDf[i,:])
    
    if doPlot:
        """
        Plotting function, but I dont know if really helpful
        """
        plt.clf()
        i=-1
        for ii,mobID in enumerate(mobTypeIDs):
            i+=2
            plt.subplot(3,2,i)
            plt.bar([x-.25 for x in catDict.keys()],earth.calShareHHTye[:,mobID],width=.75, color='g')
            plt.bar(catDict.keys(),simulationDf[:,ii],width=.5)
            plt.ylabel(mobDict[mobID])
        plt.xticks(list(catDict.keys()), list(catDict.values()),  rotation=30)
        
        plt.subplot(1,2,2)
        plt.pcolormesh(errorDf[:])
        plt.colorbar()
        plt.ylim([0,11])
        plt.xticks([.5,1.5,2.5], [mobDict[mobID] for mobID in mobTypeIDs],  rotation=30)
        plt.yticks([x-1 for x in list(catDict.keys())], list(catDict.values()),  rotation=60)
        plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.1)
    
    
    # normalize weights
    weightOfCat = weightOfCat / np.sum(weightOfCat)
    
    return errorDf, earth.calShareHHTye, simulationDf, weightOfCat

def assess_km_share_income(earth, ):
    """
    Method to assess the share of km per mobility type classified bye person income
    """
    enums = earth.getEnums() 
    mobDict = earth.getEnums()['mobilityTypes'].copy()
    mobTypeIDs = [0, 2, 4]
    if not hasattr(earth, 'calShareIncome'):
        #dfShareAge = pd.read_csv( 'resources/calData_age_km_share_2008.csv', index_col=1, header=1)
        dfShareIncome = pd.read_csv( 'resources/calData_income_km_share_2008_new.csv', index_col=[0,1], header=0)
        earth.calShareIncome = dfShareIncome
    
    # ordered dictionary for income categories
    catDict = OrderedDict()
    #populate dict
    [catDict.update({key: classStr}) for classStr,key in zip(*earth.calShareIncome.index.levels)]
    
    # fitler dictionary to use with the earth filer function
    incomeFilter = OrderedDict([
            (1 , lambda a : (a<500)),
            (2 , lambda a : (a>=500) & (a < 2000)),
            (3 , lambda a : (a>=2000) & (a < 4000)),
            (4 , lambda a : (a>=4000) & (a < 6000)),
            (5 , lambda a : (a>=6000))])

    # vector of assumed mean distances per trip
    # trips categories are 
    # 0-500m, 500m - 2.5km, 2.5-10km, 10-50km and < 50km
    MEAN_KM_PER_TRIP = [.25, 3., 7.5, 30., 75. ]
    

    errorDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])
    simulationDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])

    # weighting array of the age categories (propotional to number of agents)
    weightOfCat = np.zeros(len(catDict))

    
    hhIDs     = earth.getAttrOfAgentType('hhID',3)
    income    = earth.getAttrOfAgents('income', hhIDs)
    mobChoice = earth.getAttrOfAgentType('mobType' ,3)
    nJourneys = earth.getAttrOfAgentType('nJourneys',  3)

    for i,cat in enumerate(incomeFilter.keys()):
        
        mobKmOfCat     = np.dot(nJourneys[incomeFilter[cat](income)],MEAN_KM_PER_TRIP)
        mobChoiceOfCat = mobChoice[incomeFilter[cat](income)]
        # weight proportial to the number of persons
        weightOfCat[i] = len(mobChoiceOfCat) 
        xx = [np.sum(mobKmOfCat[mobChoiceOfCat==mobOpt]) for mobOpt in mobTypeIDs]
        simulationDf.loc[cat]  = xx / np.sum(xx)        
        errorDf.loc[cat]  = np.abs(earth.calShareIncome.iloc[i,:] - simulationDf.iloc[i,:])
    
    # normalize weights
    weightOfCat = weightOfCat / np.sum(weightOfCat)
    
    return errorDf, earth.calShareIncome, simulationDf, weightOfCat


def assess_km_share_county(earth, ):
    enums = earth.getEnums() 
    mobDict = earth.getEnums()['mobilityTypes'].copy()
    mobTypeIDs = [0, 2, 4]
    
    # vector of assumed mean distances per trip
    # trips categories are 
    # 0-500m, 500m - 2.5km, 2.5-10km, 10-50km and < 50km
    MEAN_KM_PER_TRIP = [.25, 3., 7.5, 30., 75. ]
    
    
    if not hasattr(earth, 'calShareCounty'):
        #dfShareAge = pd.read_csv( 'resources/calData_age_km_share_2008.csv', index_col=1, header=1)
        dfShareCounty = pd.read_csv( 'resources/calData_county_km_share_2008.csv', index_col=[0,1], header=0)
        earth.calShareCounty = dfShareCounty

    regionMapping = OrderedDict([
            (1 , lambda a : a['regionId']==1520), #hamburg
            (2 , lambda a : a['regionId']==6321), #bremen
            (3 , lambda a : a['regionId']==1518)])#niedersachsen
            
    errorDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])
    simulationDf = pd.DataFrame([], columns=[enums['mobilityTypes'][x] for x in mobTypeIDs])

    # weighting array of the age categories (propotional to number of agents)
    weightOfCat = np.zeros(len(regionMapping))
    
    peopleDict = OrderedDict()
    for i,cat in enumerate(regionMapping.keys()):

        #get region IDs of all persons
        cells = earth.getAgentsByFilteredType(regionMapping[cat],en.CELL)
        peopleDict[cat] = []
        for cell in cells:
            peopleDict[cat].extend(cell.peList)
        
        mobChoiceOfCat = earth.getAttrOfAgents('mobType' ,peopleDict[cat])
        nJourneys = earth.getAttrOfAgents('nJourneys',  peopleDict[cat])
        
        if nJourneys is not None:
            mobKmOfCat     = np.dot(nJourneys,MEAN_KM_PER_TRIP)
        else:
            mobKmOfCat = [0.]*5
            mobChoiceOfCat = []
        # weight proportial to the number of persons
        weightOfCat[i] = len(mobChoiceOfCat) 
        xx = [np.sum(mobKmOfCat[mobChoiceOfCat==mobOpt]) for mobOpt in mobTypeIDs]
        simulationDf.loc[cat]  = xx / np.sum(xx)        
        errorDf.loc[cat]  = np.abs(earth.calShareIncome.iloc[i,:] - simulationDf.iloc[i,:])

    # normalize weights
    weightOfCat = weightOfCat / np.sum(weightOfCat)
    
    return errorDf, earth.calShareCounty, simulationDf, weightOfCat        

def reComputePreferences(earth):
    """
    Method to change the preference of persons
    """
    
    en = core.enum #enumeration
    rad = earth.getParameters()['radicality']
    
    #re-init of the option class. Import this file again after changes done
    opGen = Opinion(earth)
    
    # loop over all households and the persons in the household to get 
    # new preferences
    i =0
    for hh in earth.getAgents.byType(agTypeID = en.HH):
        nKids  = hh.attr['nKids']
        nPers  = hh.attr['hhSize']
        income = hh.attr['income']
        for pers in hh.adults:
            i+=1
            pers.attr['preferences'] = opGen.getPref(pers.attr['age'],pers.attr['gender'],nKids,nPers,income, rad)
    print(str(i) + ' persons updated')
    
def scatterPrefVSProp(earth):
    """ 
    Function to create serveral scatter plot to visualize the dependence
    between the agents properties (or household) and the agents priorities
    """
    import matplotlib.pyplot as plt
    
    xList  = []
    
    xList.append(earth.getAttrOfAgentType('age',3))
    tmp = earth.getAttrOfAgentType('gender',3).astype(float)
    tmp += np.random.random(len(tmp))*.1
    xList.append(tmp)
    
    hhIDs = earth.getAttrOfAgentType('hhID',3)
    xList.append(earth.getAttrOfAgents('income', hhIDs)) 
    
    xLabel = ['age', 'gender', 'income']
    
    yList = []
    
    for i in range(4):
        yList.append(earth.getAttrOfAgentType('preferences',3)[:,i])
    yLabel = list(earth.getEnums()['priorities'].values())
    #%%
    plt.clf()
    ii= 0
    for i in range(len(xList)):
        for j in range(len(yList)):    
            ii+=1
            ax = plt.subplot(len(xList), len(yList),ii)
            
            plt.scatter(xList[i], yList[j])
            plt.legend()
            #if j == 0:
            plt.ylabel(yLabel[j])
            #if i==len(xList)-1:
            plt.xlabel(xLabel[i])
            ax.autoscale(tight=True)            
    plt.tight_layout()

def computeError(earth):
    errorList = []
    err, _, _, weig = assess_km_share_age(earth)
    weightedError = (err.sum(axis=1)*weig).mean()
    errorList.append(weightedError)
    print('Error in age classification       ' + str(weightedError))
    err, _, _, weig = assess_km_share_hh_type(earth)
    weightedError = (err.sum(axis=1)*weig).mean()
    errorList.append(weightedError)
    print('Error in househole classification ' + str(weightedError))
    err, _, _, weig = assess_km_share_income(earth)
    weightedError = (err.sum(axis=1)*weig).mean()
    errorList.append(weightedError)
    print('Error in income classification    ' + str(weightedError))   
    err, _, _, weig = assess_km_share_county(earth)
    weightedError = (err.sum(axis=1)*weig).mean()
    errorList.append(weightedError)
    print('Error in county classification    ' + str(weightedError))   
    totalError = np.sum(errorList)
    print('===========================================')
    print('Total error                       ' + str(totalError))   
    return errorList, totalError
   
def workflow(earth):
    
    computeError(earth)
    # calls the recompute function to get updated preferences
    reComputePreferences(earth)
    #scatterPrefVSProp(earth)

    for i in range(10):
        earth.fakeStep()
#        [plt.plot(i, xx,'x') for xx in x]
    
    computeError(earth)
    
import random
class Opinion():
    """
    Creates preferences for households, given their properties
    ToDO:
        - Update the method + more sophisticate method maybe using the DLR Data
    """
    def __init__(self, world):
        self.charAge            = world.getParameters()['charAge']
        self.indiRatio          = world.getParameters()['individualPrio']
        self.minIncomeEco       = world.getParameters()['minIncomeEco']
        self.convIncomeFraction = world.getParameters()['charIncome']

    def getPref(self, age, sex, nKids, nPers, income, radicality):


        # priority of ecology
        ce = 0.5
        if sex == 2:
            ce +=1.5
        if income > self.minIncomeEco:
            rn = random.random()
            if random.random() > 0.9:
                ce += 3.
            elif rn > 0.6:
                ce += 2.
        elif income > 2 * self.minIncomeEco:
            if random.random() > 0.8:
                ce+=4.


        ce = float(ce)**2

        # priority of convinience
        cc = 0
        cc += nKids
        cc += income/self.convIncomeFraction/2
        if sex == 1:
            cc +=1

        cc += 2* float(age)/self.charAge
        cc = float(cc)**2

        # priority of money
        cm = 0
        cm += 2* self.convIncomeFraction/income * nPers
        cm += nPers
        cm = float(cm)**2


        sumC = cc + ce + cm
        #cc /= sumC
        #ce /= sumC
        #cs /= sumC
        #cm /= sumC

        # priority of innovation
        if sex == 1:
            if income>self.minIncomeEco:
                ci = random.random()*5
            else:
                ci = random.random()*2
        else:
            if income>self.minIncomeEco:
                ci = random.random()*3
            else:
                ci = random.random()*1
        
        ci += (50. /age)**2                
        ci = float(ci)**2

        # normalization
        sumC = cc +  ce + cm +ci
        cc /= sumC
        ce /= sumC
        #cs /= sumC
        cm /= sumC
        ci /= sumC

        #individual preferences
        cci, cei,  cmi, cii = np.random.rand(4)
        sumC = cci + cei + + cmi + cii
        cci /= sumC
        cei /= sumC
        #csi /= sumC
        cmi /= sumC
        cii /= sumC

        #csAll = cs* (1-self.indiRatio) + csi*self.indiRatio
        ceAll = ce* (1-self.indiRatio) + cei*self.indiRatio
        ccAll = cc* (1-self.indiRatio) + cci*self.indiRatio
        cmAll = cm* (1-self.indiRatio) + cmi*self.indiRatio
        ciAll = ci* (1-self.indiRatio) + cii*self.indiRatio

        pref = np.asarray([ ccAll, ceAll, cmAll, ciAll])
        pref = pref ** radicality
        pref = pref / np.sum(pref)

        assert all (pref > 0) and all (pref < 1) ##OPTPRODUCTION
        
        return tuple(pref)
    