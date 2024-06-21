# -*- coding: utf-8 -*-
"""
Created in July 2023, last update: June 2024

Purpose
-------
    A 0D chemostat model with microbes in marine OMZs,
    Modular denitrification included, yields of denitrifiers depend on Gibbs free energy.
    
@authors: 
    Xin Sun, Emily Zakem, Pearse Buchanan
"""

#%% imports
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten

# numerical packages
from numba import jit

#%% Set initial conditions and incoming concentrations to chemostat experiment

### Organic matter (S)
## Constant supply (assuming 1 m3 box and flux into top of box)
Sd0_exp = np.arange(0.1,11.1,0.1) #np.arange(0.1,11.1,2) #for a quicker test
### pulse conditions  
xpulse_Sd = 0

### Oxygen supply (µM/d)
O20_exp = np.arange(0,10.1,0.2) #np.arange(0,10.1,2) #for a quicker test

### model parameters for running experiments 
### dil = 0 for checking N balance 
dil = 0.04  # dilution rate (1/day)
if dil == 0:
    days = 10  # number of days to run chemostat
    dt = 0.001  # timesteps per day (days)
    timesteps = days/dt     # number of timesteps
    out_at_day = 0.1       # output results this often (days)
    nn_output = days/out_at_day     # number of entries for output
    print("dilution = 0, check N balance")
else:
    days = 1e4  # number of days to run chemostat
    dt = 0.001  # timesteps length (days)
    timesteps = days/dt     # number of timesteps
    out_at_day = dt         # output results this often
    nn_output = days/out_at_day     # number of entries for output
    print("dilution > 0, run experiments")
nn_outputforaverage = int(2000/out_at_day) # finish value is the average of the last XX (number) of outputs
     
#%% Define variables 
outputd1 = Sd0_exp
outputd2 = O20_exp


#%% initialize arrays for output

# Nutrients
fin_O2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_Sd = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO3 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NH4 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2O = np.ones((len(outputd1), len(outputd2))) * np.nan 
# Biomasses
fin_bHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bHetC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b1DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b2DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b3DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b4DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b5DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b6DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bAOX = np.ones((len(outputd1), len(outputd2))) * np.nan
# Growth rates
fin_uHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uHetC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u1DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u2DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u3DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u4DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u5DenC = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u6DenC = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uAOX = np.ones((len(outputd1), len(outputd2))) * np.nan
# Rates 
fin_rHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rHetAer = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rO2C = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rAOX = np.ones((len(outputd1), len(outputd2))) * np.nan

#%% set traits of the different biomasses
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean")
### set traits
from traits import * 
fname = 'ConstantOM_' 

#%% calculate R*-stars
from O2_star_Xin import O2_star
from N2O_star_Xin import N2O_star 
from R_star_Xin import R_star


# O2 (nM-O2) 
O2_star_aer = R_star(dil, K_o2_aer, mumax_Het / y_oO2, y_oO2) * 1e3 
O2_star_aoo = R_star(dil, K_o2_aoo, mumax_AOO / y_oAOO, y_oAOO) * 1e3 
O2_star_noo = R_star(dil, K_o2_noo, mumax_NOO / y_oNOO, y_oNOO) * 1e3 
# N2O (nM-N)
N2O_star_den5 = R_star(dil, K_n2o_Den, VmaxN_5Den, y_n5N2O) * 1e3
# OM
OM_star_aer = R_star(dil, K_s, VmaxS, y_oHet)
OM_star_aerC = R_star(dil, K_sC, VmaxSC, y_oHet)
OM_star_den1 = R_star(dil, K_s, VmaxS, y_n1Den)
OM_star_den2 = R_star(dil, K_s, VmaxS, y_n2Den)
OM_star_den3 = R_star(dil, K_s, VmaxS, y_n3Den)
OM_star_den4 = R_star(dil, K_s, VmaxS, y_n4Den)
OM_star_den5 = R_star(dil, K_s, VmaxS, y_n5Den)
OM_star_den6 = R_star(dil, K_s, VmaxS, y_n6Den)
OM_star_den1C = R_star(dil, K_sC, VmaxSC, y_n1Den)
OM_star_den2C = R_star(dil, K_sC, VmaxSC, y_n2Den)
OM_star_den3C = R_star(dil, K_sC, VmaxSC, y_n3Den)
OM_star_den4C = R_star(dil, K_sC, VmaxSC, y_n4Den)
OM_star_den5C = R_star(dil, K_sC, VmaxSC, y_n5Den)
OM_star_den6C = R_star(dil, K_sC, VmaxSC, y_n6Den)
# Ammonia
Amm_star_aoo = R_star(dil, K_n_AOO, VmaxN_AOO, y_nAOO)
Amm_star_aox = R_star(dil, K_nh4_AOX, VmaxNH4_AOX, y_nh4AOX)
# Nitrite
nitrite_star_den2 = R_star(dil, K_n_Den, VmaxN_2Den, y_n2NO2)
nitrite_star_den4 = R_star(dil, K_n_Den, VmaxN_4Den, y_n4NO2)
nitrite_star_noo = R_star(dil, K_n_NOO, VmaxN_NOO, y_nNOO)
nitrite_star_aox = R_star(dil, K_no2_AOX, VmaxNO2_AOX, y_no2AOX)
# Nitrate
nitrate_star_den1 = R_star(dil, K_n_Den, VmaxN_1Den, y_n1NO3)
nitrate_star_den3 = R_star(dil, K_n_Den, VmaxN_3Den, y_n3NO3)
nitrate_star_den6 = R_star(dil, K_n_Den, VmaxN_6Den, y_n6NO3)


#%% begin loop of experiments
from model import OMZredox

for k in np.arange(len(outputd1)):
    for m in np.arange(len(O20_exp)):
        print(k,m)
        
        # 1) Chemostat influxes (µM-N or µM O2)
        in_Sd = Sd0_exp[k]
        in_O2 = O20_exp[m]
        in_NO3 = 30
        in_NO2 = 0.0
        in_NH4 = 0.0
        in_N2 = 0.0
        in_N2O = 0.0
        # initial conditions
        initialOM = in_Sd
        initialNO2 = 0
        
        # 2) Initial biomasses (set to 0.0 to exclude a microbial group, 0.1 as default)        
        in_bHet = 0.1
        in_b1Den = 0.1 # NO3-->NO2, cross-feed
        in_b4Den = 0.1 # NO2-->N2O, cross-feed
        in_b5Den = 0.1 # N2O-->N2, cross-feed
        in_b2Den = 0.1 # NO2-->N2
        in_b3Den = 0.1 # Complete denitrifier
        in_b6Den = 0.1 # NO3-->N2O
        in_b7Den = 0#.1 # bookend: NO3-->NO2, N2O-->N2
        # all copiotrophic heterotrophs
        in_copio = 0#.1
        in_bHetC = in_copio
        in_b1DenC = in_copio # NO3-->NO2, cross-feed
        in_b4DenC = in_copio # N2O producer (NO2-->N2O), cross-feed
        in_b5DenC = in_copio # N2O consumer (N2O-->N2), cross-feed
        in_b2DenC = in_copio # NO2-->N2
        in_b3DenC = in_copio # complete denitrifier
        in_b6DenC = in_copio # N2O producer (NO3-->N2O)
        
        in_bAOO = 0.1
        in_bNOO = 0.1
        in_bAOX = 0.1
        
        # pulse conditions        
        pulse_int = 50 #default
        pulse_Sd = xpulse_Sd #pulse intensity
        pulse_O2 = 0.0
        
        
        # 3) Call main model
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
                           pulse_Sd, pulse_O2, pulse_int, \
                           K_o2_aer, K_o2_aoo, K_o2_noo, \
                           K_n2o_Den, \
                           mumax_Het, mumax_AOO, mumax_NOO, mumax_AOX, \
                           VmaxS, K_s, VmaxSC, K_sC, \
                           VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_4Den, VmaxN_5Den, VmaxN_6Den, K_n_Den, \
                           VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
                           VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
                           y_oHet, y_oO2, \
                           y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_n4Den, y_n4NO2, y_n5Den, y_n5N2O, y_n6Den, y_n6NO3, y_n7Den_NO3, y_n7NO3, e_n7Den_NO3, y_n7Den_N2O, y_n7N2O, e_n7Den_N2O,\
                           y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
                           e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, e_n4Den, e_n5Den, e_n6Den, e_n1Den, \
                           initialOM, initialNO2, in_Sd, in_O2, in_NO3, in_NO2, in_NH4, in_N2, in_N2O, \
                           in_bHet, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX, in_b4Den, in_b5Den, in_b6Den, in_b7Den,\
                           in_bHetC, in_b1DenC, in_b2DenC, in_b3DenC, in_b4DenC, in_b5DenC, in_b6DenC)
        
        out_Sd = results[0]
        out_O2 = results[1]
        out_NO3 = results[2]
        out_NO2 = results[3]
        out_NH4 = results[4]
        out_N2O = results[5] 
        out_N2 = results[6]
        out_bHet = results[7]
        out_b1Den = results[8]
        out_b2Den = results[9]
        out_b3Den = results[10]
        out_b4Den = results[11]
        out_b5Den = results[12]
        out_b6Den = results[13]
        out_bHetC = results[14]
        out_b1DenC = results[15]
        out_b2DenC = results[16]
        out_b3DenC = results[17]
        out_b4DenC = results[18]
        out_b5DenC = results[19]
        out_b6DenC = results[20]
        out_bAOO = results[21]
        out_bNOO = results[22]
        out_bAOX = results[23]
        out_uHet = results[24]
        out_u1Den = results[25]
        out_u2Den = results[26]
        out_u3Den = results[27]
        out_u4Den = results[28]
        out_u5Den = results[29]
        out_u6Den = results[30]
        out_uHetC = results[31]
        out_u1DenC = results[32]
        out_u2DenC = results[33]
        out_u3DenC = results[34]
        out_u4DenC = results[35]
        out_u5DenC = results[36]
        out_u6DenC = results[37]      
        out_uAOO = results[38]
        out_uNOO = results[39]
        out_uAOX = results[40]
        out_rHet = results[41]
        out_rHetAer = results[42]
        out_rO2C = results[43]
        out_r1Den = results[44]
        out_r2Den = results[45]
        out_r3Den = results[46]
        out_r4Den = results[47]
        out_r5Den = results[48]
        out_r6Den = results[49]
        out_rAOO = results[50]
        out_rNOO = results[51]
        out_rAOX = results[52]     
        out_b7Den = results[53]
        out_u7Den = results[54]
    
        # 4) Record solutions in initialised arrays
        fin_O2[k,m] = np.nanmean(out_O2[-nn_outputforaverage::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-nn_outputforaverage::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-nn_outputforaverage::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-nn_outputforaverage::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-nn_outputforaverage::])
        fin_N2[k,m] = np.nanmean(out_N2[-nn_outputforaverage::])
        fin_N2O[k,m] = np.nanmean(out_N2O[-nn_outputforaverage::]) 
        fin_bHet[k,m] = np.nanmean(out_bHet[-nn_outputforaverage::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-nn_outputforaverage::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-nn_outputforaverage::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-nn_outputforaverage::])
        fin_b4Den[k,m] = np.nanmean(out_b4Den[-nn_outputforaverage::]) 
        fin_b5Den[k,m] = np.nanmean(out_b5Den[-nn_outputforaverage::]) 
        fin_b6Den[k,m] = np.nanmean(out_b6Den[-nn_outputforaverage::])
        fin_b7Den[k,m] = np.nanmean(out_b7Den[-nn_outputforaverage::]) 
        fin_bHetC[k,m] = np.nanmean(out_bHetC[-nn_outputforaverage::])
        fin_b1DenC[k,m] = np.nanmean(out_b1DenC[-nn_outputforaverage::])
        fin_b2DenC[k,m] = np.nanmean(out_b2DenC[-nn_outputforaverage::])
        fin_b3DenC[k,m] = np.nanmean(out_b3DenC[-nn_outputforaverage::])
        fin_b4DenC[k,m] = np.nanmean(out_b4DenC[-nn_outputforaverage::]) 
        fin_b5DenC[k,m] = np.nanmean(out_b5DenC[-nn_outputforaverage::]) 
        fin_b6DenC[k,m] = np.nanmean(out_b6DenC[-nn_outputforaverage::]) 
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-nn_outputforaverage::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-nn_outputforaverage::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-nn_outputforaverage::])
        fin_uHet[k,m] = np.nanmean(out_uHet[-nn_outputforaverage::])
        fin_u1Den[k,m] = np.nanmean(out_u1Den[-nn_outputforaverage::])
        fin_u2Den[k,m] = np.nanmean(out_u2Den[-nn_outputforaverage::])
        fin_u3Den[k,m] = np.nanmean(out_u3Den[-nn_outputforaverage::])
        fin_u4Den[k,m] = np.nanmean(out_u4Den[-nn_outputforaverage::]) 
        fin_u5Den[k,m] = np.nanmean(out_u5Den[-nn_outputforaverage::]) 
        fin_u6Den[k,m] = np.nanmean(out_u6Den[-nn_outputforaverage::]) 
        fin_u7Den[k,m] = np.nanmean(out_u7Den[-nn_outputforaverage::])
        fin_uHetC[k,m] = np.nanmean(out_uHetC[-nn_outputforaverage::])
        fin_u1DenC[k,m] = np.nanmean(out_u1DenC[-nn_outputforaverage::])
        fin_u2DenC[k,m] = np.nanmean(out_u2DenC[-nn_outputforaverage::])
        fin_u3DenC[k,m] = np.nanmean(out_u3DenC[-nn_outputforaverage::])
        fin_u4DenC[k,m] = np.nanmean(out_u4DenC[-nn_outputforaverage::]) 
        fin_u5DenC[k,m] = np.nanmean(out_u5DenC[-nn_outputforaverage::]) 
        fin_u6DenC[k,m] = np.nanmean(out_u6DenC[-nn_outputforaverage::]) 
        fin_uAOO[k,m] = np.nanmean(out_uAOO[-nn_outputforaverage::])
        fin_uNOO[k,m] = np.nanmean(out_uNOO[-nn_outputforaverage::])
        fin_uAOX[k,m] = np.nanmean(out_uAOX[-nn_outputforaverage::])
        fin_rHet[k,m] = np.nanmean(out_rHet[-nn_outputforaverage::])
        fin_rHetAer[k,m] = np.nanmean(out_rHetAer[-nn_outputforaverage::])
        fin_rO2C[k,m] = np.nanmean(out_rO2C[-nn_outputforaverage::])
        fin_r1Den[k,m] = np.nanmean(out_r1Den[-nn_outputforaverage::])
        fin_r2Den[k,m] = np.nanmean(out_r2Den[-nn_outputforaverage::])
        fin_r3Den[k,m] = np.nanmean(out_r3Den[-nn_outputforaverage::])
        fin_r4Den[k,m] = np.nanmean(out_r4Den[-nn_outputforaverage::]) 
        fin_r5Den[k,m] = np.nanmean(out_r5Den[-nn_outputforaverage::]) 
        fin_r6Den[k,m] = np.nanmean(out_r6Den[-nn_outputforaverage::]) 
        fin_rAOO[k,m] = np.nanmean(out_rAOO[-nn_outputforaverage::])
        fin_rNOO[k,m] = np.nanmean(out_rNOO[-nn_outputforaverage::])
        fin_rAOX[k,m] = np.nanmean(out_rAOX[-nn_outputforaverage::])

# delete results only save fin (average)
del results
del out_Sd, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_N2O
del out_bHet, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den, out_b7Den, out_bAOO, out_bNOO, out_bAOX
del out_bHetC, out_b1DenC, out_b2DenC, out_b3DenC, out_b4DenC, out_b5DenC, out_b6DenC
del out_uHet, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den, out_u7Den, out_uAOO, out_uNOO, out_uAOX
del out_uHetC, out_u1DenC, out_u2DenC, out_u3DenC, out_u4DenC, out_u5DenC, out_u6DenC,
del out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rAOX

#%% Plots

#%% Fig2_a/b_constantOM
fstic = 16
fslab = 18
fsleg = 10.5
colmap = lighten(cmo.haline, 0.8)
## [(1)]
# fig = plt.figure(figsize=(4.5,10))
## [(2)]
fig = plt.figure(figsize=(4,10))
gs = GridSpec(3, 1)

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('')

OMx = (Sd0_exp * dil) / (30*dil)



plt.plot(OMx, fin_b1Den[:,0], '-', color='firebrick', label='NO$_3$$^-$→NO$_2$$^-$', linewidth = 3)
plt.plot(OMx, fin_b6Den[:,0], '--', color='firebrick', label='NO$_3$$^-$→N$_2$O', linewidth = 3)
plt.plot(OMx, fin_b3Den[:,0], ':', color='firebrick', label='NO$_3$$^-$→N$_2$', linewidth = 3)
## [(1)]
# plt.axvline(x = 1/(y_n1Den/y_n1NO3), color='grey', linestyle='-', label='', linewidth = 3)
# plt.axvline(x = 1/(y_n6Den/y_n6NO3), color='grey', linestyle='--', label='', linewidth = 3)
# plt.axvline(x = 1/(y_n3Den/y_n3NO3), color='grey', linestyle=':', label='', linewidth = 3)
## [(2)]
plt.plot(OMx, fin_b4Den[:,0], '-', color='goldenrod', label='NO$_2$$^-$→N$_2$O', linewidth = 3)
plt.plot(OMx, fin_b2Den[:,0], '--', color='goldenrod', label='NO$_2$$^-$→N$_2$', linewidth = 3)
plt.plot(OMx, fin_b5Den[:,0], '-', color='royalblue', label='N$_2$O→N$_2$', linewidth = 3) 


plt.legend(loc='lower right', fontsize=10)


# creat subplots
ax3 = plt.subplot(gs[1,0])
ax3.set_title('')
plt.plot(OMx, (fin_r2Den[:,0] + fin_r3Den[:,0] + fin_r4Den[:,0] + fin_r6Den[:,0]) * 1e3, '-', color='black', linewidth = 3)

ax2=ax3.twinx()
ax2.plot(OMx, (fin_b1Den[:,0]*1 + fin_b6Den[:,0]*2 + fin_b3Den[:,0] *3 + fin_b4Den[:,0]*1 + fin_b2Den[:,0]*2 + fin_b5Den[:,0] *1)/(fin_b1Den[:,0] + fin_b6Den[:,0] + fin_b3Den[:,0] + fin_b4Den[:,0] + fin_b2Den[:,0] + fin_b5Den[:,0]), '-', color='forestgreen', label='mean pathway length', linewidth = 3)


# creat subplots
ax4 = plt.subplot(gs[2,0])
ax4.set_title('')
plt.plot(OMx, fin_NO3[:,0], '-', color='black', label='eq NO$_3$$^-$', linewidth = 3)

ax5=ax4.twinx()
plt.plot(OMx, fin_Sd[:,0], '-', color='forestgreen', label='eq OM', linewidth = 3)


## [(1)]
# ax1.set_ylabel('Biomass (µM-N)', fontsize=fslab)
# ax3.set_ylabel('N loss via den (nM-N/d)', fontsize=fslab) 
# ax4.set_ylabel('Nitrate (µM)', fontsize=fslab) 
# ax1.tick_params(labelbottom=False, labelsize=fstic)
# ax2.tick_params(labelbottom=False, labelright=False, labelsize=fstic)
# ax4.tick_params(labelsize=fstic)
# ax3.tick_params(labelbottom=False, labelsize=fstic)
# ax5.tick_params(labelright=False, labelsize=fstic)
## [(2)]
ax1.tick_params(labelbottom=False, labelleft=False, labelsize=fstic)
ax2.tick_params(labelbottom=False, labelright=False, labelsize=fstic)
ax3.tick_params(labelbottom=False, labelleft=False, labelsize=fstic)
ax4.tick_params(labelleft=False, labelsize=fstic)
ax5.tick_params(labelright=False, labelsize=fstic)

ax2.tick_params(axis='y', labelcolor='forestgreen')
ax5.tick_params(axis='y', labelcolor='forestgreen')
ax4.set_xlabel('OM:NO$_3$$^-$ supply', fontsize=fslab)

xlowerlimit = 0
xupperlimit = 0.365
xtickdiv = 0.1
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

ax1.set_ylim([0.0, 1.0])
ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
ax2.set_ylim([1.0, 3.5])
ax2.set_yticks(np.arange(1.0, 3.55, 0.5))
ax3.set_ylim([0, 1250])
ax3.set_yticks(np.arange(0, 1250, 200))
ax4.set_ylim([-0.5, 30.5])
ax4.set_yticks(np.arange(0, 30.5, 5))
ax5.set_ylim([-0.2, 7.2])
ax5.set_yticks(np.arange(0, 7.2, 1))

plt.tight_layout()

#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('Fig2_a/b.png', dpi=300)

#%% FigS_steadystate (OMpulse=0), denitrifier coexistence
fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(5,3.5))
gs = GridSpec(1, 1)

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('', fontsize=fslab)
OMx=O20_exp

plt.plot(OMx, fin_b1Den[0,:], '-', color='firebrick', label='NO$_3$$^-$-->NO$_2$$^-$')
plt.plot(OMx, fin_b6Den[0,:], '--', color='firebrick', label='NO$_3$$^-$-->N$_2$O')
plt.plot(OMx, fin_b3Den[0,:], ':', color='firebrick', label='NO$_3$$^-$-->N$_2$')
plt.plot(OMx, fin_b4Den[0,:], '-', color='goldenrod', label='NO$_2$$^-$-->N$_2$O')
plt.plot(OMx, fin_b2Den[0,:], '--', color='goldenrod', label='NO$_2$$^-$-->N$_2$')
plt.plot(OMx, fin_b5Den[0,:], '-', color='royalblue', label='N$_2$O-->N$_2$') 

plt.legend(loc='center right', fontsize=8.5)
ax1.set_xlabel('O$_2$ supply (µM/d)', fontsize=fslab)
ax1.set_ylabel('Biomass (µM-N)', fontsize=fslab)

plt.tight_layout()

#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('FigS1_SteadyStateDenCoexist.png', dpi=300)     

#%% save the output to data folder
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/Output")

fname = 'OMconstant_'
np.savetxt(fname+'_O2supply.txt', O20_exp, delimiter='\t')
np.savetxt(fname+'_O2.txt', fin_O2, delimiter='\t')
np.savetxt(fname+'_N2.txt', fin_N2, delimiter='\t')
np.savetxt(fname+'_N2O.txt', fin_N2O, delimiter='\t')
np.savetxt(fname+'_NO3.txt', fin_NO3, delimiter='\t')
np.savetxt(fname+'_NO2.txt', fin_NO2, delimiter='\t')
np.savetxt(fname+'_NH4.txt', fin_NH4, delimiter='\t')
np.savetxt(fname+'_OM.txt', fin_Sd, delimiter='\t')

np.savetxt(fname+'_bHet.txt', fin_bHet, delimiter='\t')
np.savetxt(fname+'_b1Den.txt', fin_b1Den, delimiter='\t')
np.savetxt(fname+'_b2Den.txt', fin_b2Den, delimiter='\t')
np.savetxt(fname+'_b3Den.txt', fin_b3Den, delimiter='\t')
np.savetxt(fname+'_b4Den.txt', fin_b4Den, delimiter='\t')
np.savetxt(fname+'_b5Den.txt', fin_b5Den, delimiter='\t')
np.savetxt(fname+'_b6Den.txt', fin_b6Den, delimiter='\t')
np.savetxt(fname+'_bAOO.txt', fin_bAOO, delimiter='\t')
np.savetxt(fname+'_bNOO.txt', fin_bNOO, delimiter='\t')
np.savetxt(fname+'_bAOX.txt', fin_bAOX, delimiter='\t')
np.savetxt(fname+'_bHetC.txt', fin_bHetC, delimiter='\t')
np.savetxt(fname+'_b1DenC.txt', fin_b1DenC, delimiter='\t')
np.savetxt(fname+'_b2DenC.txt', fin_b2DenC, delimiter='\t')
np.savetxt(fname+'_b3DenC.txt', fin_b3DenC, delimiter='\t')
np.savetxt(fname+'_b4DenC.txt', fin_b4DenC, delimiter='\t')
np.savetxt(fname+'_b5DenC.txt', fin_b5DenC, delimiter='\t')
np.savetxt(fname+'_b6DenC.txt', fin_b6DenC, delimiter='\t')

np.savetxt(fname+'_uHet.txt', fin_uHet, delimiter='\t')
np.savetxt(fname+'_u1Den.txt', fin_u1Den, delimiter='\t')
np.savetxt(fname+'_u2Den.txt', fin_u2Den, delimiter='\t')
np.savetxt(fname+'_u3Den.txt', fin_u3Den, delimiter='\t')
np.savetxt(fname+'_u4Den.txt', fin_u4Den, delimiter='\t')
np.savetxt(fname+'_u5Den.txt', fin_u5Den, delimiter='\t')
np.savetxt(fname+'_u6Den.txt', fin_u6Den, delimiter='\t')
np.savetxt(fname+'_uAOO.txt', fin_uAOO, delimiter='\t')
np.savetxt(fname+'_uNOO.txt', fin_uNOO, delimiter='\t')
np.savetxt(fname+'_uAOX.txt', fin_uAOX, delimiter='\t')
np.savetxt(fname+'_uHetC.txt', fin_uHetC, delimiter='\t')
np.savetxt(fname+'_u1DenC.txt', fin_u1DenC, delimiter='\t')
np.savetxt(fname+'_u2DenC.txt', fin_u2DenC, delimiter='\t')
np.savetxt(fname+'_u3DenC.txt', fin_u3DenC, delimiter='\t')
np.savetxt(fname+'_u4DenC.txt', fin_u4DenC, delimiter='\t')
np.savetxt(fname+'_u5DenC.txt', fin_u5DenC, delimiter='\t')
np.savetxt(fname+'_u6DenC.txt', fin_u6DenC, delimiter='\t')

np.savetxt(fname+'_rHet.txt', fin_rHet, delimiter='\t')
np.savetxt(fname+'_rHetAer.txt', fin_rHetAer, delimiter='\t')
np.savetxt(fname+'_r1Den.txt', fin_r1Den, delimiter='\t')
np.savetxt(fname+'_r2Den.txt', fin_r2Den, delimiter='\t')
np.savetxt(fname+'_r3Den.txt', fin_r3Den, delimiter='\t')
np.savetxt(fname+'_r4Den.txt', fin_r4Den, delimiter='\t')
np.savetxt(fname+'_r5Den.txt', fin_r5Den, delimiter='\t')
np.savetxt(fname+'_r6Den.txt', fin_r6Den, delimiter='\t')
np.savetxt(fname+'_rAOO.txt', fin_rAOO, delimiter='\t')
np.savetxt(fname+'_rNOO.txt', fin_rNOO, delimiter='\t')
np.savetxt(fname+'_rAOX.txt', fin_rAOX, delimiter='\t')
np.savetxt(fname+'_rO2C.txt', fin_rO2C, delimiter='\t')