#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:10:06 2019

@author: balderrama
"""

import pandas as pd
from joblib import load
import numpy as np

Data = pd.read_csv('bo-1_0_0_0_0_0_0_0_0.csv',index_col=0)    
Data['HouseHolds'] = Data['Pop2025']/Data['NumPeoplePerHH']
filename = 'Regressions/demand_regression.joblib'
lm_D = load(filename) 
HouseHolds = np.array(list(Data['HouseHolds']))
HouseHolds = HouseHolds.reshape(-1, 1)


Techno_Economic = pd.read_excel('Regressions/Tecno_economic_parameters.xlsx',index_col=0,Header=None)
Data['Total Fuel Cost'] = (0.8 + (2 * 0.8 * 33.7 * Data['TravelHours'] )/15000)     

X = pd.DataFrame(index=range(len(Data)))
X.loc[:,'LLP'] = Techno_Economic['Value']['LLP']
X.loc[:,'Renewable invesment cost'] = Techno_Economic['Value']['PV Cost']
X.loc[:,'Genererator invesment cost'] = Techno_Economic['Value']['Gen Cost']
X.loc[:,'Battery invesment cost'] = Techno_Economic['Value']['Bat Cost']
X.loc[:,'Demand'] =  lm_D.predict(HouseHolds)/(1 - 0.05)
X.loc[:,'Diesel Cost'] = list(Data['Total Fuel Cost'])
X.loc[:,'Solar energy'] = list(Data['GHI'])

filename = 'Regressions/LCOE_regression.joblib'
lm_LCOE = load(filename)   
filename = 'Regressions/NPC_regression.joblib'
lm_NPC = load(filename)  

Y = X.copy()
      
Y.loc[:,'LCOE Generation'] = lm_LCOE.predict(X)


filename = 'Regressions/demand_regression_Actualize.joblib'
lm_D_A = load(filename)
Y.loc[:,'Demand Actualize'] = lm_D_A.predict(HouseHolds)/(1 - 0.05)
Y.loc[:,'Hybrid_Transmission_Cost']= list(Data['Hybrid_Transmission_Cost'])
Y.loc[:,'LCOE Transmisition'] = (Y['Hybrid_Transmission_Cost']/Y['Demand Actualize'])*1000
Y.loc[:, 'LCOE'] = Y['LCOE Transmisition'] + Y['LCOE Generation']
Y.loc[:,'NPC Generation'] = lm_NPC.predict(X)
Y.loc[:,'NPC'] =  Y['NPC Generation'] + Y['Hybrid_Transmission_Cost']
       
 