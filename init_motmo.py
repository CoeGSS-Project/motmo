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
#%% IMPORT 

import sys, os, socket
import matplotlib as plt
import csv
import time
import pprint
import logging as lg
import numpy as np
import pandas as pd
from scipy import signal


sys.path.append('../gcfabm')
from classes_motmo import Person, GhostPerson, Household, GhostHousehold, Cell, GhostCell, Earth, Opinion
from gcfabm import core
import scenarios

print('import done')

#%%
comm = core.comm
mpiRank = core.mpiRank
mpiSize = core.mpiSize


en = core.enum



if not socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    plt.use('Agg')
###### Enums ################
#connections
CON_CC = 1 # loc - loc
CON_CH = 2 # loc - household
CON_HH = 3 # household, household
CON_HP = 4 # household, person
CON_PP = 5 # person, person

#nodes
CELL   = 1
HH     = 2
PERS   = 3

#time spans
_month = 1
_year  = 2

en.CON_LL = 1 # loc - loc
en.CON_LH = 2 # loc - household
en.CON_HH = 3 # household, household
en.CON_HP = 4 # household, person
en.CON_PP = 5 # person, person

#nodes
en.CELL   = 1
en.HH     = 2
en.PERS   = 3


# Mobility setup setup
def mobilitySetup(earth):
    parameters = earth.getParameters()

    # define convenience functions
    def convenienceBrown(density, pa, kappa, cell):
        
        conv = pa['minConvB'] +\
        kappa * (pa['maxConvB'] - pa['minConvB']) * \
        np.exp( - (density - pa['muConvB'])**2 / (2 * pa['sigmaConvB']**2))
        return conv

    def convenienceGreen(density, pa, kappa, cell):
        
        conv = pa['minConvG'] + \
        (pa['maxConvGInit']-pa['minConvG']) * \
        (1 - kappa)  * np.exp( - (density - pa['muConvGInit'])**2 / (2 * pa['sigmaConvGInit']**2)) +  \
        (pa['maxConvG'] - pa['minConvG']) * kappa * (np.exp(-(density - pa['muConvG'])**2 / (2 * pa['sigmaConvB']**2)))
        
        
        return conv


    def conveniencePublicLeuphana(density, pa, kappa, cell):
        
        currKappa   = (1 - kappa) * pa['maxConvGInit'] + kappa * pa['maxConvG']
        return pa['conveniencePublic'][cell._node['coord']] * currKappa


    def conveniencePublic(density, pa, kappa, cell):
        conv = pa['minConvP'] + \
        ((pa['maxConvP'] - pa['minConvP']) * (kappa)) * \
        np.exp(-(density - pa['muConvP'])**2 / (2 * ((1 - kappa) * \
                   pa['sigmaConvPInit'] + (kappa * pa['sigmaConvP']))**2))
        
        return conv
    
    def convenienceShared(density, pa, kappa, cell):
#        conv = pa['minConvS'] + \
#        ((pa['maxConvS'] - pa['minConvS']) * (kappa)) * \
#        np.exp(-(density - pa['muConvS'])**2 / (2 * ((1 - kappa) * \
#                   pa['sigmaConvSInit'] + (kappa * pa['sigmaConvS']))**2))
        
        conv = (kappa/10.) + pa['minConvS'] + (kappa *(pa['maxConvS'] - pa['minConvS'] - (kappa/10.))  +\
                    ((1-kappa)* (pa['maxConvSInit'] - pa['minConvS'] - (kappa/10.)))) * \
                    np.exp( - (density - pa['muConvS'])**2 / (2 * ((1-kappa) * \
                    pa['sigmaConvSInit'] + (kappa * pa['sigmaConvS']))**2) )        
        return conv
        
    def convenienceNone(density, pa, kappa, cell):
        conv = pa['minConvN'] + \
        ((pa['maxConvN'] - pa['minConvN']) * (kappa)) * \
        np.exp(-(density - pa['muConvN'])**2 / (2 * ((1 - kappa) * \
                   pa['sigmaConvNInit'] + (kappa * pa['sigmaConvN']))**2))        
        return conv


    from collections import OrderedDict
    
    # register brown:
    propDict = OrderedDict()
    propDict['emissions']      = parameters['initEmBrown'] 
    propDict['fixedCosts']     = parameters['initPriceBrown']
    propDict['operatingCosts'] = parameters['initOperatingCostsBrown']
    
    earth.registerGood('brown',                                # name
                    propDict,                                  # (emissions, TCO)
                    convenienceBrown,                          # convenience function
                    'start',                                   # time step of introduction in simulation
                    initExperience = parameters['techExpBrown'], # initial experience
                    priceRed = parameters['priceReductionB'],  # exponent for price reduction through learning by doing
                    emRed    = parameters['emReductionB'],     # exponent for emission reduction through learning by doing
                    emFactor = parameters['emFactorB'],        # factor for emission reduction through learning by doing
                    emLimit  = parameters['emLimitB'],         # emission limit
                    weight   = parameters['weightB'])          # average weight

    # register green:
    propDict = OrderedDict()
    propDict['emissions']      = parameters['initEmGreen'] 
    propDict['fixedCosts']     = parameters['initPriceGreen']
    propDict['operatingCosts'] = parameters['initOperatingCostsGreen']    
  
    earth.registerGood('green',                                #name
                    propDict,                                  # (emissions, TCO)
                    convenienceGreen,                          # convenience function
                    'start',
                    initExperience = parameters['techExpGreen'],                # initial experience
                    priceRed = parameters['priceReductionG'],  # exponent for price reduction through learning by doing
                    emRed    = parameters['emReductionG'],     # exponent for emission reduction through learning by doing
                    emFactor = parameters['emFactorG'],        # factor for emission reduction through learning by doing
                    emLimit  = parameters['emLimitG'],         # emission limit
                    weight   = parameters['weightG'])          # average weight

    # register public:
    propDict = OrderedDict()
    if parameters['scenario'] == 2:
        convFuncPublic = conveniencePublicLeuphana
    else:
        convFuncPublic = conveniencePublic
    propDict['emissions']      = parameters['initEmPublic'] 
    propDict['fixedCosts']     = parameters['initPricePublic']
    propDict['operatingCosts'] = parameters['initOperatingCostsPublic']
 
    earth.registerGood('public',  #name
                    propDict,   #(emissions, TCO)
                    convFuncPublic,
                    'start',
                    initExperience = parameters['techExpPublic'],
                    pt2030  = parameters['pt2030'],          # emissions 2030 (compared to 2012)
                    ptLimit = parameters['ptLimit'])         # emissions limit (compared to 2012)

                    
    # register shared:
    propDict = OrderedDict()
    propDict['emissions']      = parameters['initEmShared']
    propDict['fixedCosts']     = parameters['initPriceShared']
    propDict['operatingCosts'] = parameters['initOperatingCostsShared']
   
    earth.registerGood('shared', # name
                    propDict,    # (emissions, TCO)
                    convenienceShared,
                    'start',
                    initExperience = parameters['techExpShared'],
                    weight = parameters['weightS'],           # average weight
                    initMaturity = parameters['initMaturityS']) # initital maturity
    # register none:    
    propDict = OrderedDict()
    propDict['emissions']      = parameters['initEmNone'] 
    propDict['fixedCosts']     = parameters['initPriceNone']
    propDict['operatingCosts'] = parameters['initOperatingCostsNone']

    earth.registerGood('none',  #name
                    propDict,   #(emissions, TCO)
                    convenienceNone,
                    'start',
                    initExperience = parameters['techExpNone'],
                    initMaturity = parameters['initMaturityN']) # initital maturity
    

    earth.setParameter('nMobTypes', len(earth.getEnums()['brands']))
    return earth
    ##############################################################################


def createAndReadParameters(fileName, dirPath):
    def readParameterFile(parameters, fileName):
        for item in csv.DictReader(open(fileName)):
            if item['name'][0] != '#':
                parameters[item['name']] = core.convertStr(item['value'])
        return parameters

    parameters = core.AttrDict()
    # reading of gerneral parameters
    parameters = readParameterFile(parameters, 'parameters_all.csv')
    # reading of scenario-specific parameters
    parameters = readParameterFile(parameters, fileName)

    lg.info('Setting loaded:')

    if mpiRank == 0:
        parameters = scenarios.create(parameters, dirPath)

        parameters = initExogeneousExperience(parameters)
        parameters = randomizeParameters(parameters)
    else:
        parameters = None
    return parameters

def exchangeParameters(parameters):
    parameters = comm.bcast(parameters, root=0)

    if mpiRank == 0:
        print('Parameter exchange done')
    lg.info('Parameter exchange done')

    return parameters

def householdSetup(earth, calibration=False):
    
    tt = time.time()
    #enumerations for h5File - second dimension
    H5NPERS  = 0
    H5AGE    = 1
    H5GENDER = 2
    H5INCOME = 3
    H5HHTYPE = 4
    H5MOBDEM = [5, 6, 7, 8, 9]
    
    
    parameters = earth.getParameters()
    tt = time.time()
    parameters['population'] = np.ceil(parameters['population'])
    nAgents = 0
    nHH     = 0
    overheadAgents = 2000 # additional agents that are loaded 
    tmp = np.unique(parameters['regionIdRaster'])
    tmp = tmp[~np.isnan(tmp)]
    regionIdxList = tmp[tmp>0]

    nRegions = np.sum(tmp>0)
   
    if earth.isParallel:
        boolMask = parameters['mpiRankLayer']== core.mpiRank
    else:
        boolMask = parameters['landLayer'].astype(np.bool)
    nAgentsOnProcess = np.zeros(nRegions)
    for i, region in enumerate(regionIdxList):
        boolMask2 = parameters['regionIdRaster']== region
        nAgentsOnProcess[i] = np.sum(parameters['population'][boolMask & boolMask2])

        if nAgentsOnProcess[i] > 0:
            # calculate start in the agent file (20 overhead for complete households)
            nAgentsOnProcess[i] += overheadAgents

    if earth.isParallel:
        nAgentsPerProcess = earth.papi.all2all(nAgentsOnProcess)
    else:
        nAgentsPerProcess = [nAgentsOnProcess]
    nAgentsOnProcess = np.array(nAgentsPerProcess)
    lg.info('Agents on process:' + str(nAgentsOnProcess))
    hhData = dict()
    currIdx = dict()
    h5Files = dict()
    import h5py
    for i, region in enumerate(regionIdxList):
        # all processes open all region files (not sure if necessary)
        lg.debug('opening file: ' + parameters['resourcePath'] + 'people' + str(int(region)) + 'new.hdf5')
        h5Files[i]      = h5py.File(parameters['resourcePath'] + 'people' + str(int(region)) + 'new.hdf5', 'r')
    
    core.mpiBarrier()

    for i, region in enumerate(regionIdxList):

        offset = 0
        agentStart = int(np.sum(nAgentsOnProcess[:core.mpiRank,i]))
        agentEnd   = int(np.sum(nAgentsOnProcess[:core.mpiRank+1,i]))

        lg.info('Reading agents from ' + str(agentStart) + ' to ' + str(agentEnd) + ' for region ' + str(region))
        lg.debug('Vertex count: ' + str(earth.graph.nCount()))
        
        if earth.debug:
            pass
            #earth.view(str(earth.papi.rank) + '.png')


        dset = h5Files[i].get('people')
        hhData[i] = dset[offset + agentStart: offset + agentEnd,]
        #print hhData[i].shape
        
        if nAgentsOnProcess[core.mpiRank, i] == 0:
            continue

        assert hhData[i].shape[0] >= nAgentsOnProcess[core.mpiRank,i] ##OPTPRODUCTION
        
        idx = 0
        # find the correct possition in file
        nPers = int(hhData[i][idx, 0])
        if np.sum(np.diff(hhData[i][idx:idx+nPers, H5NPERS])) !=0:

            #new index for start of a complete household
            idx = idx + np.where(np.diff(hhData[i][idx:idx+nPers, H5NPERS]) != 0)[0][0]
        currIdx[i] = int(idx)


    core.mpiBarrier() # all loading done

    for i, region in enumerate(regionIdxList):
        h5Files[i].close()
    lg.info('Loading agents from file done')

    opinion     = Opinion(earth)
    nAgentsCell = 0
    locDict = earth.getLocationDict()
    
    
    for x, y in list(locDict.keys()):
        
        nAgentsCell = int(parameters['population'][x, y]) + nAgentsCell # subtracting Agents that are places too much in the last cell
        loc         = earth.grid.getLocation(x, y)
        region      = parameters['regionIdRaster'][x, y]
        regionIdx   = np.where(regionIdxList == region)[0][0]

        while 1:
            successFlag = False
            nPers   = int(hhData[regionIdx][currIdx[regionIdx], H5NPERS])
            #print nPers,'-',nAgents
            ages    = hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5AGE]
            genders = hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5GENDER]
            
            nAdults = np.sum(ages>= 18)
            
            nKids = np.sum(ages < 18)
            
            if nAdults == 0:
                currIdx[regionIdx]  += nPers
                lg.info('Household without adults skipped')
                continue
                
            if currIdx[regionIdx] + nPers > hhData[regionIdx].shape[0]:
                print('Region: ' + str(regionIdxList[regionIdx]))
                print('asked size: ' + str(currIdx[regionIdx] + nPers))
                print('hhDate shape: ' + str(hhData[regionIdx].shape))

            income = hhData[regionIdx][currIdx[regionIdx], H5INCOME]
            hhType = hhData[regionIdx][currIdx[regionIdx], H5HHTYPE]

            
            #income *= (1.- (0.1 * max(3, nKids))) # reduction fo effective income by kids
            # calculate actual income from equqlized income
            # see: http://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Equivalised_income
            income *= 1 + (np.sum(ages>= 14)-1) * .5 + np.sum(ages< 14) *.3
            # to monthly income
            income /= 12.
            # set minimal income
            income = max(400., income)
            #income *= parameters['mobIncomeShare'] 


            nJourneysPerPerson = hhData[regionIdx][currIdx[regionIdx]:currIdx[regionIdx]+nPers, H5MOBDEM]


            # creating houshold
            
            hh = Household(earth,
                           coord=(x, y),
                           hhSize=nPers,
                           nKids=nKids,
                           income=income,
                           expUtil=0,
                           util=0,
                           expenses=0,
                           hhType=hhType)

            hh.adults = list()
            hh.register(earth, parentEntity=loc, liTypeID=CON_CH)
            

            hh.loc['population'] = hh.loc['population'] + nPers
#            if earth.isParallel:
            hhID = hh.attr['gID']
#            else:
#                hhID = hh.nID
            assert nAdults > 0 ##OPTPRODUCTION
            
            for iPers in range(nPers):

                nAgentsCell -= 1
                nAgents     += 1

                if ages[iPers] < 18:
                    continue    #skip kids
                prefTuple = opinion.getPref(ages[iPers], genders[iPers], nKids, nPers, income, parameters['radicality'])

                
                assert len(nJourneysPerPerson[iPers]) == 5##OPTPRODUCTION
                pers = Person(earth,
                              preferences = np.asarray(prefTuple),
                              hhID        = hhID,
                              gender      = genders[iPers],
                              age         = ages[iPers],
                              nJourneys   = nJourneysPerPerson[iPers],
                              util        = 0.,
                              commUtil    = np.asarray([0.5, 0.1, 0.4, 0.3, 0.1]), # [0.5]*parameters['nMobTypes'],
                              selfUtil    = np.asarray([np.nan]*parameters['nMobTypes']),
                              mobType     = 0,
                              prop        = np.asarray([0.]*len(parameters['properties'])),
                              consequences= np.asarray([0.]*len(prefTuple)),
                              lastAction  = 0,
                              hhType      = hhType,
                              emissions   = 0.)
                
                pers.imitation = np.random.randint(parameters['nMobTypes'])
                pers.register(earth, parentEntity=hh, liTypeID=CON_HP)
                
                successFlag = True
            
            
            currIdx[regionIdx]  += nPers
            nHH                 += 1
            if not successFlag:
                import pdb
                pdb.set_trace()
            if nAgentsCell <= 0:
                break
    lg.info('All agents initialized in ' + "{:2.4f}".format((time.time()-tt)) + ' s')

    if earth.isParallel:
        earth.papi.transferGhostAgents(earth)

        
    for hh in earth.getAgentsByType(HH, ghosts = False):                  ##OPTPRODUCTION
        assert len(hh.adults) == hh['hhSize'] - hh['nKids']  ##OPTPRODUCTION
        
    core.mpiBarrier()
    lg.info(str(nAgents) + ' Agents and ' + str(nHH) +
            ' Housholds created in -- ' + str(time.time() - tt) + ' s')
    
    if mpiRank == 0:
        print('Household setup done')
            
    return earth




def initEarth(simNo,
              outPath,
              parameters,
              maxNodes,
              maxLinks,
              debug,
              mpiComm=None):
    tt = time.time()
    
    
    earth = Earth(simNo,
                  outPath,
                  parameters,
                  maxNodes=maxNodes,
                  maxLinks=maxLinks,
                  debug=debug,
                  mpiComm=mpiComm,
                  agentOutput= parameters['writeAgentFile'],
                  linkOutput = parameters['writeLinkFile'] )

    #global assignment
    core.earth = earth
    
    lg.info('Init finished after -- ' + str( time.time() - tt) + ' s')
    if mpiRank == 0:
        print('Earth init done')
    return earth

def initScenario(earth, parameters):

    ttt = time.time()
    earth.initMarket(earth,
                     parameters.properties,
                     parameters.randomCarPropDeviationSTD,
                     burnIn=parameters.burnIn)

    earth.market.mean = np.array([400., 300.])
    earth.market.std  = np.array([100., 50.])
    
    earth.market.initExogenousExperience(parameters['scenario'])

    
    #init location memory
    enums = earth.getEnums()
    enums['priorities'] =   {0: 'convinience',
                             1: 'ecology',
                             2: 'money',
                             3: 'innovation'}


    enums['properties'] =   {1: 'emissions',
                             2: 'fixedCosts',
                             3: 'operatingCosts'}

    enums['agTypeIDs'] =   {1: 'cell',
                            2: 'household',
                            3: 'pers'}

    enums['consequences'] =   {0: 'convenience',
                               1: 'eco-friendliness',
                               2: 'remaining money',
                               3: 'innovation'}

    enums['mobilityTypes'] =   {0: 'combustion',
                                1: 'electic',
                                2: 'public',
                                3: 'shared',
                                4: 'none'}

    enums['hhTypes'] =   {      1: '1Adulds_young',
                                2: '1Adulds_medium',
                                3: '1Adulds_elderly',
                                4: '2Adulds_young',
                                5: '2Adulds_medium',
                                6: '2Adulds_elderly',
                                7: '3+Adultes',
                                8: 'HH+kids',
                                9: 'HH+teen',
                                10:'HH+minor',
                                11:'SingelParent'}

    initTypes(earth)
    
    initSpatialLayer(earth)
    
    initInfrastructure(earth)
    
    mobilitySetup(earth)
    
    cellTest(earth)
    
    initGlobalRecords(earth)
    
    householdSetup(earth)
    
    generateNetwork(earth)
    
    initMobilityTypes(earth)
    
    if parameters['writeAgentFile']:
        initAgentOutput(earth)
    
    if parameters['writeLinkFile']:
        initLinkOutput(earth)
        
    lg.info('Init of scenario finished after -- ' + "{:2.4f}".format((time.time()-ttt)) + ' s')
    if mpiRank == 0:
        print('Scenario init done in ' + "{:2.4f}".format((time.time()-ttt)) + ' s')
    return earth   



def initTypes(earth):
    tt = time.time()

    en.CELL = earth.registerAgentType(AgentClass=Cell, GhostAgentClass= GhostCell,
                               staticProperties  = [('gID', np.int32, 1),
                                                   ('coord', np.int16, 2),
                                                   ('regionId', np.int16, 1),
                                                   ('popDensity', np.float64, 1),
                                                   ('population', np.int16, 1)],
                               dynamicProperties = [('convenience', np.float64, 5),
                                                   ('carsInCell', np.int32, 5),
                                                   ('chargStat', np.int32, 1),
                                                   ('emissions', np.float64, 5),
                                                   ('electricConsumption', np.float64, 1)])

    en.HH = earth.registerAgentType(AgentClass=Household, 
                                GhostAgentClass=GhostHousehold,
                                staticProperties  = [('gID', np.int32, 1),
                                                    ('coord', np.int16, 2),
                                                    ('hhSize', np.int8,1),
                                                    ('nKids', np.int8, 1),
                                                    ('hhType', np.int8, 1)],
                                dynamicProperties = [('income', np.float64, 1),
                                                    ('expUtil', np.float64, 1),
                                                    ('util', np.float64, 1),
                                                    ('expenses', np.float64, 1)])


    en.PERS = earth.registerAgentType(AgentClass=Person, GhostAgentClass= GhostPerson,
                                staticProperties = [('gID', np.int32, 1),
                                                   ('hhID', np.int32, 1),
                                                   ('preferences', np.float64, 4),
                                                   ('gender', np.int8, 1),
                                                   ('nJourneys', np.int16, 5),
                                                   ('hhType', np.int8, 1)],
                               dynamicProperties = [('age', np.int8, 1),
                                                   ('util', np.float64, 1),     # current utility
                                                   ('commUtil', np.float64, 5), # comunity utility
                                                   ('selfUtil', np.float64, 5), # own utility at time of action
                                                   ('mobType', np.int8, 1),
                                                   ('prop', np.float64, 3),
                                                   ('consequences', np.float64, 4),
                                                   ('lastAction', np.int16, 1),
                                                   ('emissions', np.float64, 1),
                                                   ('costs', np.float64, 1)])


    en.CON_CC = earth.registerLinkType('cell-cell', CELL, CELL, staticProperties = [('weig', np.float64, 1)])
    en.CON_CH = earth.registerLinkType('cell-hh', CELL, HH)
    en.CON_HH = earth.registerLinkType('hh-hh', HH,HH)
    en.CON_HP = earth.registerLinkType('hh-pers', HH, PERS)
    en.CON_PP = earth.registerLinkType('pers-pers', PERS, PERS, dynamicProperties = [('weig', np.float64,1)])

    if mpiRank == 0:
        print('Initialization of types done in ' + "{:2.4f}".format((time.time()-tt)) + ' s')

    return CELL, HH, PERS

def initSpatialLayer(earth):
    earth.registerGrid(CELL, CON_CC)
    tt = time.time()
    parameters = earth.getParameters()
    connList= earth.grid.computeConnectionList(parameters['connRadius'], ownWeight=1.5)
    #print(parameters['landLayer'])
    #print(parameters['mpiRankLayer'])
    if earth.isParallel:
        earth.grid.init((parameters['landLayer']),
                               connList, 
                               LocClassObject=Cell,
                               mpiRankArray=parameters['mpiRankLayer'])
    else:
        earth.grid.init((parameters['landLayer']),
                               connList, 
                               LocClassObject=Cell)
    convMat = np.asarray([[0., 1, 0.],[1., 1., 1.],[0., 1., 0.]])
    tmp = parameters['population']*parameters['reductionFactor']
    tmp[np.isnan(tmp)] = 0
    smoothedPopulation = signal.convolve2d(tmp,convMat,boundary='symm',mode='same')
    tmp = parameters['cellSizeMap']
    tmp[np.isnan(tmp)] = 0
    smoothedCellSize   = signal.convolve2d(tmp,convMat,boundary='symm',mode='same')

    
    popDensity = np.divide(smoothedPopulation, 
                           smoothedCellSize, 
                           out=np.zeros_like(smoothedPopulation), 
                           where=smoothedCellSize!=0)
    popDensity[popDensity>4000.]  = 4000.
    
   
    if 'regionIdRaster' in list(parameters.keys()):

        for cell in earth.random.shuffleAgentsOfType(CELL):
            cell.attr['regionId'] = parameters['regionIdRaster'][tuple(cell['coord'])]
            cell.attr['chargStat'] = 0
            cell.attr['emissions'] = np.zeros(len(earth.getEnums()['mobilityTypes']))
            cell.attr['electricConsumption'] = 0.
            cell.cellSize = parameters['cellSizeMap'][tuple(cell['coord'])]
            cell.attr['popDensity'] = popDensity[tuple(cell['coord'])]
    if earth.isParallel:        
        earth.papi.updateGhostAgents([CELL],['chargStat'])

    if mpiRank == 0:
        print('Setup of the spatial layer done in'  + "{:2.4f}".format((time.time()-tt)) + ' s')
        
        
def initInfrastructure(earth):
    tt = time.time()
    # infrastructure
    earth.initChargInfrastructure()
    
    if mpiRank == 0:
        print('Infrastructure setup done in ' + "{:2.4f}".format((time.time()-tt)) + ' s')

#%% cell convenience test
def cellTest(earth):

    for good in list(earth.market.goods.values()):
        
        good.emissionFunction(good, earth.market)
        #good.updateMaturity()  
        
    nLocations = len(earth.getLocationDict())
    convArray  = np.zeros([earth.market.getNMobTypes(), nLocations])
    popArray   = np.zeros(nLocations)
    eConvArray = earth.para['landLayer'] * 0
    
    #import tqdm
    if earth.para['showFigures']:
        for i, cell in enumerate(earth.random.shuffleAgentsOfType(CELL)):        
            #tt = time.time()
            convAll, popDensity = cell.selfTest(earth)
            convAll[1] = convAll[1] * cell.electricInfrastructure(100.)
            convArray[:, i] = convAll
            popArray[i] = popDensity
            eConvArray[cell['coord']] = convAll[1]
            #print time.time() - ttclass
        
        
            
        plt.figure('electric infrastructure convenience')
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(eConvArray)
        plt.title('el. convenience')
        plt.clim([-.2,np.nanmax(eConvArray)])
        plt.colorbar()
#        plt.subplot(2,2,2)
#        plt.imshow(earth.para['chargStat'])
        plt.clim([-2,10])
        plt.title('number of charging stations')
        plt.colorbar()
        
        plt.subplot(2,2,3)
        plt.imshow(earth.para['landLayer'])
        #plt.clim([-2,10])
        plt.title('processes ID')
        
        plt.subplot(2,2,4)
        plt.imshow(earth.para['population'])
        #plt.clim([-2,10])
        plt.title('population')
        
        
        plt.figure()
        for i in range(earth.market.getNMobTypes()):
            if earth.market.getNMobTypes() > 4:
                plt.subplot(3, int(np.ceil(earth.market.getNMobTypes()/3.)), i+1)
            else:
                plt.subplot(2, 2, i+1)
            plt.scatter(popArray,convArray[i,:], s=2)
            plt.title('convenience of ' + earth.getEnums()['mobilityTypes'][i])
        plt.show()
        adsf
        
# %% Generate Network
def generateNetwork(earth):
    tt = time.time()
    parameters = earth.getParameters()
    
    tt = time.time()
    
    earth.generateSocialNetwork(PERS,CON_PP)
    
    lg.info( 'Social network initialized in -- ' + str( time.time() - tt) + ' s')
    if parameters['scenario'] == 0 and earth.para['showFigures']:
        earth.view(str(earth.papi.rank) + '.png')
    if mpiRank == 0:
        print('Social network setup done in ' + "{:2.4f}".format((time.time()-tt)) + ' s')
        #core.plotGraph(earth, PERS, CON_PP)

def initMobilityTypes(earth):
    tt = time.time()
    earth.market.initialCarInit()
    earth.market.setInitialStatistics([1000.,2.,350., 100.,50.])
    for goodKey in list(earth.market.goods.keys()):##OPTPRODUCTION
        #print earth.market.goods[goodKey].properties.keys() 
        #print earth.market.properties
        assert list(earth.market.goods[goodKey].properties.keys()) == earth.market.properties ##OPTPRODUCTION
    
    if mpiRank == 0:
        print('Setup of mobility types done in '  + "{:2.4f}".format((time.time()-tt)) + ' s')


def initGlobalRecords(earth):
    tt = time.time()
    parameters = earth.getParameters()

    calDataDfCV = pd.read_csv(parameters['resourcePath'] + 'calDataCV.csv', index_col=0, header=0)
    calDataDfEV = pd.read_csv(parameters['resourcePath'] + 'calDataEV.csv', index_col=0, header=0)


    enums = earth.getEnums()
    for re in parameters['regionIDList']:
        earth.registerRecord('stock_' + str(re),
                         'total use per mobility type -' + str(re),
                         list(enums['mobilityTypes'].values()),
                         style='plot',
                         mpiReduce='sum')
        
        earth.registerRecord('elDemand_' + str(re),
                         'electric Demand -' + str(re),
                         ['electric_demand'],
                         style='plot',
                         mpiReduce='sum')
        
        earth.registerRecord('emissions_' + str(re),
                         'co2Emissions -' + str(re),
                         list(enums['mobilityTypes'].values()),
                         style='plot',
                         mpiReduce='sum')

        earth.registerRecord('nChargStations_' + str(re),
                         'Number of charging stations -' + str(re),
                         ['nChargStations'],
                         style='plot',
                         mpiReduce='sum')

        timeIdxs = list()
        values   = list()

        for column in calDataDfCV.columns[1:]:
            value = [np.nan]*earth.para['nMobTypes'] # value[0]: combustion cars, value[1]: electric cars
            timeIdx = earth.year2step(int(column))            
            value[0] = (calDataDfCV[column]['re_' + str(re)] ) 
            # for electric cars we have data for less years then for the combustion cars, so we can
            # use the columns loop of the combustion cars
            if column in calDataDfEV.columns[1:]:
                value[1] = (calDataDfEV[column]['re_' + str(re)] ) 


            timeIdxs.append(timeIdx)
            values.append(value)

        earth.globalRecord['stock_' + str(re)].addCalibrationData(timeIdxs,values)

    earth.registerRecord('growthRate', 'Growth rate of mobitlity types',
                         list(enums['mobilityTypes'].values()), style='plot')
    earth.registerRecord('allTimeProduced', 'Overall production of car types',
                         list(enums['mobilityTypes'].values()), style='plot')
    earth.registerRecord('maturities', 'Technological maturity of mobility types',
                         ['mat_B', 'mat_G', 'mat_P', 'mat_S', 'mat_N'], style='plot')
    earth.registerRecord('globEmmAndPrice', 'Properties',
                         ['meanEmm','stdEmm','meanFiC','stdFiC','meanOpC','stdOpC'], style='plot')

    if mpiRank == 0:
        print('Setup of global records done in '  + "{:2.4f}".format((time.time()-tt)) + ' s')

def initAgentOutput(earth):
    tt = time.time()
    #%% Init of agent file
    tt = time.time()
    core.mpiBarrier()
    lg.info( 'Waited for Barrier for ' + str( time.time() - tt) + ' s')
    tt = time.time()
    #earth.initAgentFile(typ = HH)
    #earth.initAgentFile(typ = PERS)
    #earth.initAgentFile(typ = CELL)
    earth.io.initAgentFile(earth, [CELL, HH, PERS])


    lg.info( 'Agent file initialized in ' + str( time.time() - tt) + ' s')

    if mpiRank == 0:
        print('Setup of agent output done in '  +str(time.time()-tt) + 's')

def initLinkOutput(earth):
    tt = time.time()
    #%% Init of agent file
    tt = time.time()
    core.mpiBarrier()
    lg.info( 'Waited for Barrier for ' + str( time.time() - tt) + ' s')
    tt = time.time()
    #earth.initAgentFile(typ = HH)
    #earth.initAgentFile(typ = PERS)
    #earth.initAgentFile(typ = CELL)
    earth.io.initLinkFile(earth, [CON_CC, CON_CH, CON_HP, CON_PP])


    lg.info( 'Agent file initialized in ' + str( time.time() - tt) + ' s')

    if mpiRank == 0:
        print('Setup of agent output done in '  +str(time.time()-tt) + 's')
        

def initCacheArrays(earth):
    
    maxFriends = earth.para['maxFriends']
    persZero = earth.getAgentsByType(PERS)[0]
    
    nUtil = persZero['commUtil'].shape[0]
    Person.cacheCommUtil = np.zeros([maxFriends+1, nUtil])
    Person.cacheUtil     = np.zeros(maxFriends+1)
    Person.cacheMobType  = np.zeros(maxFriends+1, dtype=np.int32)
    Person.cacheWeights  = np.zeros(maxFriends+1)
#    persZero._setSharedArrays( maxFriends, nUtil)
#    persZero._setSharedArray('cacheCommUtil', np.zeros([maxFriends+1, nUtil]))
#    persZero._setSharedArray('cacheUtil',  np.zeros(maxFriends+1))
#    persZero._setSharedArray('cacheMobType', np.zeros(maxFriends+1, dtype=np.int32))
#    persZero._setSharedArray('cacheWeights', np.zeros(maxFriends+1))
    
def initExogeneousExperience(parameters):
    inputFromGlobal         = pd.read_csv(parameters['resourcePath'] + 'inputFromGlobal.csv')
    randomFactor = (5*np.random.randn() + 100.)/100
    parameters['experienceWorldGreen']  = inputFromGlobal['expWorldGreen'].values / 10. * randomFactor
    randomFactor = (5*np.random.randn() + 100.)/100
    parameters['experienceWorldBrown']  = inputFromGlobal['expWorldBrown'].values * randomFactor
    experienceGer                       = inputFromGlobal['expGer'].values
    experienceGerGreen                  = inputFromGlobal['expGerGreen'].values
    parameters['experienceGerGreen']    = experienceGerGreen
    parameters['experienceGerBrown']    = [experienceGer[i]-experienceGerGreen[i] for i in range(len(experienceGer))]
    
    return parameters

def randomizeParameters(parameters):
    #%%
    def randDeviation(percent, minDev=-np.inf, maxDev=np.inf):
        while True:
            dev = np.random.randn() *percent
            if dev < maxDev and dev > minDev:
                break
        return (100. + dev) / 100.
    
    maxFriendsRand  = int( parameters['maxFriends'] * randDeviation(20)) 
    if maxFriendsRand > parameters['minFriends']+1:
        parameters['maxFriends'] = maxFriendsRand
    minFriendsRand  = int( parameters['minFriends'] * randDeviation(5)) 
    if minFriendsRand < parameters['maxFriends']-1:
        parameters['minFriends'] = minFriendsRand
    parameters['mobIncomeShare'] * randDeviation(5)
    parameters['charIncome'] * randDeviation(5)
    parameters['priceRedBCorrection'] * randDeviation(3, -3, 3)
    parameters['priceRedGCorrection'] * randDeviation(3, -3, 3)
    
    parameters['hhAcceptFactor'] = 1.0 + (np.random.rand()*5. / 100.) #correct later
    return parameters

    #%%
def runModel(earth, parameters):
    # %% run of the model ################################################
    lg.info('####### Running model with paramertes: #########################')
    lg.info(pprint.pformat(parameters.toDict()))
    if mpiRank == 0:
        fidPara = open(earth.para['outPath'] + '/parameters.txt', 'w')
        pprint.pprint(parameters.toDict(), fidPara)
        fidPara.close()
    lg.info('################################################################')
    
    
    #%% Initial actions
    tt = time.time()
    for household in earth.random.shuffleAgentsOfType(HH):

        household.takeActions(earth, household.adults, np.random.randint(0, earth.market.getNMobTypes(), len(household.adults)))
        for adult in household.adults:
            adult.attr['lastAction'] =  np.int(np.random.rand() * np.float(earth.para['mobNewPeriod']))

    lg.info('Initial actions done')

    for cell in earth.random.shuffleAgentsOfType(CELL):
        cell.step(earth.para, earth.market.getCurrentMaturity())
     
    
    earth.market.initPrices()
    for good in list(earth.market.goods.values()):
        good.initMaturity()
        good.updateEmissionsAndMaturity(earth.market)
        #good.updateMaturity()
    
    lg.info('Initial market step done')

    for household in earth.random.shuffleAgentsOfType(HH):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility(earth, actionTaken=True)
        #household.shareExperience(earth)
        
        
    lg.info('Initial actions randomized in -- ' + str( time.time() - tt) + ' s')

    initCacheArrays(earth)
    
    #plotIncomePerNetwork(earth)


    #%% Simulation
    earth.time = -1 # hot bugfix to have both models running #TODO Fix later
    lg.info( "Starting the simulation:")
    for step in range(parameters.nSteps):

        earth.step() # looping over all cells
        if earth.date[1] == 2008:
            break
        #plt.figure()
        #plot.calGreenNeigbourhoodShareDist(earth)
        #plt.show()




    #%% Finishing the simulation
    lg.info( "Finalizing the simulation (No." + str(earth.simNo) +"):")
    earth.saveParameters()  
    earth.saveEnumerations()
    earth.finalize()

def writeSummary(earth, parameters):
    if core.mpiRank == 0:
        fid = open(earth.para['outPath'] + '/summary.out','w')
    
        fid.writelines('Parameters:')
    
        errorTot = 0
        for re in earth.para['regionIDList']:
            error = earth.globalRecord['stock_' + str(re)].evaluateRelativeError()
            fid.writelines('Error - ' + str(re) + ': ' + str(error) + '\n')
            errorTot += error
    
        fid.writelines('Total relative error:' + str(errorTot))
    
        errorTot = np.zeros(earth.para['nMobTypes'])
        for re in earth.para['regionIDList']:
            error = earth.globalRecord['stock_' + str(re)].evaluateNormalizedError()
            fid.writelines('Error - ' + str(re) + ': ' + str(error) + '\n')
            errorTot += error
    
        fid.writelines('Total relative error:' + str(errorTot))
        
        fid.close()
    
        lg.info( 'The simulation error is: ' + str(errorTot) )

def onlinePostProcessing(earth):
    # calculate the mean and standart deviation of priorities
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']


    lg.info('Preferences -average')
    lg.info(df.mean())
    lg.info('Preferences - standart deviation')
    lg.info(df.std())

    lg.info( 'Preferences - standart deviation within friends')
    avgStd= np.zeros([1, 4])
    for agent in earth.random.shuffleAgentsOfType(HH):
        friendList = agent.getPeerIDs(liTypeID=CON_HH)
        if len(friendList) > 1:
            #print df.ix[friendList].std()
            avgStd += df.ix[friendList].std().values
    nAgents    = np.nansum(parameters.population)
    lg.info(avgStd / nAgents)
    prfType = np.argmax(df.values,axis=1)
    #for i, agent in enumerate(earth.iterNode(HH)):
    #    print agent.prefTyp, prfType[i]
    df['ref'] = prfType

    # calculate the correlation between weights and differences in priorities
    if False:
        pref = np.zeros([earth.graph.vcount(), 4])
        pref[earth.nodeDict[PERS],:] = np.array(earth.graph.vs[earth.nodeDict[PERS]]['preferences'])
        idx = list()
        for link in earth.iterLinks(CON_PP):
            link['prefDiff'] = np.sum(np.abs(pref[link.target, :] - pref[link.source,:]))
            idx.append(link.index)


        plt.figure()
        plt.scatter(np.asarray(earth.graph.es['prefDiff'])[idx],np.asarray(earth.graph.es['weig'])[idx])
        plt.xlabel('difference in preferences')
        plt.ylabel('connections weight')

        plt.show()
        x = np.asarray(earth.graph.es['prefDiff'])[idx].astype(np.float)
        y = np.asarray(earth.graph.es['weig'])[idx].astype(np.float)
        lg.info( np.corrcoef(x,y))


