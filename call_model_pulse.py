# -*- coding: utf-8 -*-
"""
Created in March 2023, update for better performance

Purpose
-------
    A 0D chemostat model with microbes in marine OMZs,
    Modular denitrification included, yields of denitrifiers depend on Gibbs free energy.
    Organic matter pulses included
    
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
Sd0_exp = 0.05 #µM-N m-3 day-1
## Pulse intensity 
xpulse_Sd = np.arange(0.1,20.1,1) #µM-N

### Oxygen supply
O20_exp =  np.arange(0,10.1,0.2)

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
    days = 5e4  # number of days to run chemostat
    dt = 0.001  # timesteps length (days)
    timesteps = days/dt     # number of timesteps
    out_at_day = dt         # output results this often
    nn_output = days/out_at_day     # number of entries for output
    print("dilution > 0, run experiments")
    
nn_outputforaverage = int(2000/out_at_day) # finish value is the average of the last XX (number) of outputs
     
#%% Define variables  
outputd1 = xpulse_Sd 
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
fname = 'OMpulse_' 

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
OM_star_den1 = R_star(dil, K_s, VmaxS, y_n1Den)
OM_star_den2 = R_star(dil, K_s, VmaxS, y_n2Den)
OM_star_den3 = R_star(dil, K_s, VmaxS, y_n3Den)
OM_star_den4 = R_star(dil, K_s, VmaxS, y_n4Den)
OM_star_den5 = R_star(dil, K_s, VmaxS, y_n5Den)
OM_star_den6 = R_star(dil, K_s, VmaxS, y_n6Den)
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
        in_Sd = Sd0_exp
        in_O2 = O20_exp[m]
        in_NO3 = 30.0
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
        in_b3Den = 0.1 # complete denitrifier
        in_b6Den = 0.1 # NO3-->N2O
        in_b7Den = 0#.1 # bookend: NO3-->NO2, N2O-->N2    
        in_bAOO = 0.1
        in_bNOO = 0.1
        in_bAOX = 0.1
        
        # pulse conditions        
        pulse_int = 50 #pulse interval
        pulse_Sd = xpulse_Sd[k]#pulse intensity
        pulse_O2 = 0.0
       
       
        # 3) Call main model
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
                           pulse_Sd, pulse_O2, pulse_int, \
                           K_o2_aer, K_o2_aoo, K_o2_noo, \
                           K_n2o_Den, \
                           mumax_Het, mumax_AOO, mumax_NOO, mumax_AOX, \
                           VmaxS, K_s, \
                           VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_4Den, VmaxN_5Den, VmaxN_6Den, K_n_Den, \
                           VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
                           VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
                           y_oHet, y_oO2, \
                           y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_n4Den, y_n4NO2, y_n5Den, y_n5N2O, y_n6Den, y_n6NO3, y_n7Den_NO3, y_n7NO3, e_n7Den_NO3, y_n7Den_N2O, y_n7N2O, e_n7Den_N2O,\
                           y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
                           e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, e_n4Den, e_n5Den, e_n6Den, e_n1Den, \
                           initialOM, initialNO2, in_Sd, in_O2, in_NO3, in_NO2, in_NH4, in_N2, in_N2O, \
                           in_bHet, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX, in_b4Den, in_b5Den, in_b6Den, in_b7Den)
        
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
        out_bAOO = results[14]
        out_bNOO = results[15]
        out_bAOX = results[16]
        out_uHet = results[17]
        out_u1Den = results[18]
        out_u2Den = results[19]
        out_u3Den = results[20]
        out_u4Den = results[21]
        out_u5Den = results[22]
        out_u6Den = results[23]
        out_uAOO = results[24]
        out_uNOO = results[25]
        out_uAOX = results[26]
        out_rHet = results[27]
        out_rHetAer = results[28]
        out_rO2C = results[29]
        out_r1Den = results[30]
        out_r2Den = results[31]
        out_r3Den = results[32]
        out_r4Den = results[33]
        out_r5Den = results[34]
        out_r6Den = results[35]
        out_rAOO = results[36]
        out_rNOO = results[37]
        out_rAOX = results[38]     
        out_b7Den = results[39]
        out_u7Den = results[40]
        
    
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
del out_uHet, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den, out_u7Den, out_uAOO, out_uNOO, out_uAOX
del out_uHetC, out_u1DenC, out_u2DenC, out_u3DenC, out_u4DenC, out_u5DenC, out_u6DenC,
del out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rAOX


#%% round up function
import math
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

#%% Plots

#%% Fig 1a_R* for 6 dens:
fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(4,5.5))

gs = GridSpec(2, 1)


ax1 = plt.subplot(gs[0,0])
## Create a list of values for the y axis
y_values = [OM_star_den1, OM_star_den6, OM_star_den3, OM_star_den4, OM_star_den2, OM_star_den5]
## Create a list of categories for the x axis
x_categories = ['NO$_3$$^-$→NO$_2$$^-$', 'NO$_3$$^-$→N$_2$O','NO$_3$$^-$→N$_2$', 'NO$_2$$^-$→N$_2$O', 'NO$_2$$^-$→N$_2$', 'N$_2$O→N$_2$']
## Create a list of colors for the bars
colors = ['teal'] * len(x_categories)

## Create a bar plot
plt.bar(x_categories, y_values, color=colors)
## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
#plt.xticks(rotation=90, fontsize=fslab)
ax1.set_ylim(ymin = 0, ymax = 0.032)
ax1.set_yticks(np.arange(0, 0.032, 0.01))
plt.ylabel('OM* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)


ax2 = plt.subplot(gs[1,0])
y_values2 = [nitrate_star_den1, nitrate_star_den6, nitrate_star_den3, nitrite_star_den4, nitrite_star_den2, N2O_star_den5*1e-3]
## Create a list of colors for the bars
colors = ['teal'] * len(x_categories)

## Create a bar plot
plt.bar(x_categories, y_values2, color=colors)
## Set the labels for the x and y axis
plt.xlabel('', fontsize=fslab)
plt.xticks(rotation=90, fontsize=fslab)

plt.ylabel('N* (µM)', fontsize=fslab)
plt.yticks(fontsize=fstic)

ax1.tick_params(labelbottom=False)

plt.tight_layout()
#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('Fig1a_Rstars.png', dpi=300)

#%% Fig2c_pulse
fstic = 16
fslab = 18
fsleg = 10.5
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(4.5,10))
gs = GridSpec(3, 1)

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('')
OMx = (xpulse_Sd/pulse_int + Sd0_exp * dil) / (30 * dil)
plt.plot(OMx, fin_b1Den[:,0], '-', color='firebrick', label='NO$_3$$^-$→NO$_2$$^-$', linewidth = 3)
plt.plot(OMx, fin_b6Den[:,0], '--', color='firebrick', label='NO$_3$$^-$→N$_2$O', linewidth = 3)
plt.plot(OMx, fin_b3Den[:,0], ':', color='firebrick', label='NO$_3$$^-$→N$_2$', linewidth = 3)
plt.plot(OMx, fin_b4Den[:,0], '-', color='goldenrod', label='NO$_2$$^-$→N$_2$O', linewidth = 3)
plt.plot(OMx, fin_b2Den[:,0], '--', color='goldenrod', label='NO$_2$$^-$→N$_2$', linewidth = 3)
plt.plot(OMx, fin_b5Den[:,0], '-', color='royalblue', label='N$_2$O→N$_2$', linewidth = 3) 

plt.legend(loc='upper left', fontsize=fsleg)

# creat subplots
ax3 = plt.subplot(gs[1,0])
ax3.set_title('')
plt.plot(OMx, (fin_r2Den[:,0] + fin_r3Den[:,0] + fin_r4Den[:,0] + fin_r6Den[:,0]) * 1e3, '-', color='black', linewidth = 3) 

ax2=ax3.twinx()
ax2.plot(OMx, (fin_b1Den[:,0]*1 + fin_b6Den[:,0]*2 + fin_b3Den[:,0] *3 + fin_b4Den[:,0]*1 + fin_b2Den[:,0]*2 + fin_b5Den[:,0] *1)/(fin_b1Den[:,0] + fin_b6Den[:,0] + fin_b3Den[:,0] + fin_b4Den[:,0] + fin_b2Den[:,0] + fin_b5Den[:,0]), '-', color='forestgreen', label='mean pathway length', linewidth = 3) 
ax2.set_ylabel('Mean pathway length', fontsize=fslab, color='forestgreen')
ax2.tick_params(axis='y', labelcolor='forestgreen')


# creat subplots
ax4 = plt.subplot(gs[2,0])
ax4.set_title('')
plt.plot(OMx, fin_NO3[:,0], '-', color='black', label='eq NO$_3$$^-$', linewidth = 3)
ax4.set_xlabel('OM:NO$_3$$^-$ supply with pulses', fontsize=fslab)
 
ax5=ax4.twinx()
plt.plot(OMx, fin_Sd[:,0], '-', color='forestgreen', label='eq OM', linewidth = 3)
ax5.set_ylabel('OM (µM-N)', fontsize=fslab, color='forestgreen')
ax5.tick_params(axis='y', labelcolor='forestgreen')

ax1.tick_params(labelbottom=False, labelleft=False, labelsize=fstic)
ax2.tick_params(labelbottom=False, labelsize=fstic)
ax3.tick_params(labelbottom=False, labelleft=False, labelsize=fstic)
ax4.tick_params(labelleft=False, labelsize=fstic)
ax5.tick_params(labelsize=fstic)

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
fig.savefig('Fig2c_pulse.png', dpi=300)


#%% Fig3_Pcolormesh_Contour

#get phi = 1
phi_y = y_oHet/y_oO2 #(O2in/OMin = yom/yo2) : phi = O2in/OMin * yo2/yom = 1
# Create a meshgrid for the calculations
phi_contourX, phi_contourY = np.meshgrid(np.arange(0,40.1,0.01) * dil, np.arange(0.1,22.1,0.02)/pulse_int + Sd0_exp * dil)
# Calculate the ratio
phi_ratio = phi_contourX/phi_contourY
phi_threshold = phi_y/1000
phi_mask = np.abs(phi_ratio - phi_y) < phi_threshold


fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(12,6)) 
gs = GridSpec(2, 3)

ax1 = plt.subplot(gs[0,0]) 
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])

contourX = outputd2 * dil
contourY = xpulse_Sd/pulse_int + Sd0_exp * dil 

colormin1 = 0.0
colormax1 = 0.8

colormin = 0.0
colormax = 600

ax1.set_title('Biomass NO$_3$$^-$→NO$_2$$^-$ (µM-N)', fontsize=fslab)
p1 = ax1.pcolormesh(contourX, contourY, fin_b1Den, vmin=colormin1, vmax=colormax1, cmap=colmap) 
ax1.plot(phi_contourX[phi_mask], phi_contourY[phi_mask], color='white')

ax2.set_title('Biomass NO$_3$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p2 = ax2.pcolormesh(contourX, contourY, fin_b6Den, vmin=colormin1, vmax=colormax1, cmap=colmap)

ax3.set_title('Biomass NO$_3$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p3 = ax3.pcolormesh(contourX, contourY, fin_b3Den, vmin=colormin1, vmax=colormax1, cmap=colmap) 

ax4.set_title('Biomass NO$_2$$^-$→N$_2$O (nM-N)', fontsize=fslab)
p4 = ax4.pcolormesh(contourX, contourY, fin_b4Den, vmin=colormin1, vmax=colormax1, cmap=colmap) 

ax5.set_title('Biomass NO$_2$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p5 = ax5.pcolormesh(contourX, contourY, fin_b2Den, vmin=colormin1, vmax=colormax1, cmap=colmap)

ax6.set_title('Biomass N$_2$O→N$_2$ (µM-N)', fontsize=fslab)
p6 = ax6.pcolormesh(contourX, contourY, fin_b5Den, vmin=colormin1, vmax=colormax1, cmap=colmap) 

## delete axis title of some subplots
ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic)
ax5.tick_params(labelsize=fstic, labelleft=False)
ax6.tick_params(labelsize=fstic, labelleft=False)

## add axis title to some subplots
contourYlabel = 'OM supply with pulses (µM-N/d)'
contourXlabel = 'O$_2$ supply (µM/d)'

ax4.set_xlabel(contourXlabel, fontsize=fslab)
ax5.set_xlabel(contourXlabel, fontsize=fslab)
ax6.set_xlabel(contourXlabel, fontsize=fslab)

#select x axis limit 
xlowerlimit = 0
xupperlimit = 10.02*dil
xtickdiv = 0.1
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]: 
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)

plt.tight_layout()

#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('Fig3_Contours.png', dpi=300)

#%% Fig3_ModelVSDataBarPlot_Rates
fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(4,3.5))
gs = GridSpec(1, 1)

# get Tracey et al., 2023 data
os.chdir("YourFolderPath/RatesDataFromJohn_2023biogeosc")
John = pd.read_csv('For_Xin_plot.csv')
John_O2 = John['Normalized_O2_across_sensors_uM']
John_NO3reduction = John['NO3_reduction_nM_NperD']
John_AMX = John['AMX_Rate_nM_NperD']
John_NO2toN2 = John['DN_Rate_nM_NperD']

# create dataframes for obs data and modeling results
Obs_John = pd.DataFrame({
    'oxygen_concentration': John_O2,
    'NO3reduction': John_NO3reduction,
    'AMX': John_AMX,
    'NO2toN2': John_NO2toN2
})

Obs_John_filtered = Obs_John[Obs_John['oxygen_concentration'] <= 2][['NO3reduction', 'NO2toN2']]


OMindexforbar = 1 #2 # default 0 if you only selected 1 OM as in put
print(xpulse_Sd[OMindexforbar])
print(xpulse_Sd[OMindexforbar]/pulse_int+Sd0_exp*dil)

Model_rates = pd.DataFrame({
    'oxygen_concentration': fin_O2[OMindexforbar],
    'NO3reduction': fin_r1Den[OMindexforbar]*1e3,
    'AMX': fin_rAOX[OMindexforbar]*1e3,
    'NO2toN2': fin_r2Den[OMindexforbar]*1e3,
    'NO3toN2O': fin_r6Den[OMindexforbar]*1e3,
    'NO3toN2': fin_r3Den[OMindexforbar]*1e3,
    'NO2toN2O': fin_r4Den[OMindexforbar]*1e3,
    'N2OtoN2': fin_r5Den[OMindexforbar]*1e3
})

Model_rates_filtered = Model_rates[Model_rates['oxygen_concentration'] <= 2][['NO3reduction', 'NO2toN2']]
# calculate mean and std
Model_rates_filtered_means = Model_rates_filtered.mean().reset_index() 
Model_rates_filtered_std = Model_rates_filtered.std().reset_index() 

Obs_John_filtered_means = Obs_John_filtered.mean().reset_index()
Obs_John_filtered_std = Obs_John_filtered.std().reset_index()


# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('', fontsize=fslab)

# plot modeling results as mean and std
x = np.array([0, 0.6]) #
width = 0.15
#plot
plt.bar(x-0.15, Model_rates_filtered_means[0], width, yerr=Model_rates_filtered_std[0], capsize=5, color = "grey") #
melted_data_model = Model_rates_filtered.melt(var_name='Category', value_name='Value')

# plot observation data
plt.bar(x, Obs_John_filtered_means[0], width, yerr=Obs_John_filtered_std[0], capsize=5,  color = "royalblue")
melted_data = Obs_John_filtered.melt(var_name='Category', value_name='Value')

ax1.set_xlabel('')
ax1.set_ylabel('Rates (nM-N/d)')
ax1.set_xticks(x-0.1)
ax1.set_xticklabels(['NO$_3$$^-$→NO$_2$$^-$', 'NO$_2$$^-$→N$_2$'])#'AMX', 

plt.legend(["Model", "Obs"]) 
ax1.set_ylabel('Rates (nM-N/d)', fontsize=fstic, color='grey')

plt.tight_layout()

#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('Fig3_bar_Rates.png', dpi=300)

#%% Fig3_ModelVSDataBarPlot_Genes
fstic = 14
fslab = 16
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(5,3.5))
gs = GridSpec(1, 1)

# get Clara's gene data
os.chdir("YourFolderPath/MicrobialDataFromClaraFuchsman")
ClaraData = pd.read_csv('DenitgeneTotalsFromClara.csv')
nar = ClaraData['nar']
nir = ClaraData['nir']
nos = ClaraData['nos']
depth = ClaraData['depth']


# create dataframes for obs data and modeling results
Obs_Clara = pd.DataFrame({
    'depth':depth,
    'nar': nar,
    'nir': nir,
    'nos': nos
})

Obs_Clara_filtered = Obs_Clara[Obs_Clara['depth'] >= 100][['nar', 'nir', 'nos']]

OMindexforbar = 1 # default 0 if you only selected 1 OM as in put
print(xpulse_Sd[OMindexforbar]/pulse_int+Sd0_exp*dil)


Model_genes = pd.DataFrame({
    'oxygen_concentration': fin_O2[OMindexforbar],
    'nar': (fin_b1Den[OMindexforbar]+fin_b3Den[OMindexforbar]+fin_b6Den[OMindexforbar])*4 *(13/12) * 1e7,
    'nar (copy=4)': (fin_b1Den[OMindexforbar]+fin_b3Den[OMindexforbar]+fin_b6Den[OMindexforbar])*4 *(13/12) * 1e7,
    'nir': (fin_b2Den[OMindexforbar]+fin_b3Den[OMindexforbar]+fin_b4Den[OMindexforbar]+fin_b6Den[OMindexforbar]) *(13/12) * 1e7,
    'nos': (fin_b2Den[OMindexforbar]+fin_b3Den[OMindexforbar]+fin_b5Den[OMindexforbar]) *(13/12) * 1e7
})

Model_genes_filtered = Model_genes[Model_genes['oxygen_concentration'] <= 2][['nar', 'nir', 'nos']] 
# calculate mean and std
Model_genes_filtered_means = Model_genes_filtered.mean().reset_index() 
Model_genes_filtered_std = Model_genes_filtered.std().reset_index() 

Obs_Clara_filtered_means = Obs_Clara_filtered.mean().reset_index()
Obs_Clara_filtered_std = Obs_Clara_filtered.std().reset_index()

# creat subplots
ax1 = plt.subplot(gs[0,0])
ax1.set_title('', fontsize=fslab)

# plot modeling results as mean and std
x = np.array([0, 1, 2])
width = 0.4
plt.bar(x-0.4, Model_genes_filtered_means[0], width, yerr=Model_genes_filtered_std[0], capsize=5, color = "grey") #
melted_data_model = Model_genes_filtered.melt(var_name='Category', value_name='Value')
ax1.tick_params(axis='y', labelcolor='grey')

# plot observation data
ax2=ax1.twinx()
plt.bar(x, Obs_Clara_filtered_means[0], width, yerr=Obs_Clara_filtered_std[0], capsize=5, color = "royalblue")
melted_data = Obs_Clara_filtered.melt(var_name='Category', value_name='Value')

ax1.set_xlabel('')
ax1.set_ylabel('Gene (copies/mL)',fontsize=fstic, color='grey')
ax1.set_xticks(x-0.1)
ax1.set_xticklabels(['nar', 'nir', 'nos'], fontstyle='italic')

ax2.set_xlabel('')
ax2.set_ylabel('Gene (normalized reads)',fontsize=fstic, color='royalblue')
ax2.set_xticks(x-0.1)
ax2.tick_params(axis='y', labelcolor='royalblue')

plt.tight_layout()

#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('Fig3_bar_Genes.png', dpi=300)

#%% FigS_Pcolormesh_Contour plot_all nuts and rates
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(10,28))
gs = GridSpec(5, 3)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])
ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])
ax10 = plt.subplot(gs[3,0])
ax11 = plt.subplot(gs[3,1])
ax12 = plt.subplot(gs[3,2])
ax13 = plt.subplot(gs[4,0])
ax14 = plt.subplot(gs[4,1])
ax15 = plt.subplot(gs[4,2])

# set titles
ax1.set_title('O$_2$ (µM)', fontsize=fslab)
ax2.set_title('NO$_3$$^-$ (µM)', fontsize=fslab)
ax3.set_title('NO$_2$$^-$ (µM)', fontsize=fslab)
ax4.set_title('OM (µM-N)', fontsize=fslab)
ax5.set_title('N$_2$O (nM)', fontsize=fslab)
ax6.set_title('Het (µM-N/d)', fontsize=fslab)
ax8.set_title('AOA (µM-N/d)', fontsize=fslab)
ax9.set_title('NOB (µM-N/d)', fontsize=fslab)
ax10.set_title('NO$_3$$^-$→NO$_2$$^-$ (µM-N/d)', fontsize=fslab)
ax11.set_title('NO$_3$$^-$→N$_2$O (µM-N/d)', fontsize=fslab)
ax12.set_title('NO$_3$$^-$→N$_2$ (µM-N/d)', fontsize=fslab)
ax13.set_title('NO$_2$$^-$→N$_2$O (µM-N/d)', fontsize=fslab)
ax14.set_title('NO$_2$$^-$→N$_2$ (µM-N/d)', fontsize=fslab)
ax15.set_title('N$_2$O→N$_2$ (µM-N/d)', fontsize=fslab)


nh4_n2_AOX = (e_n2AOX*0.5*y_nh4AOX)

# set x and y axes
contourX = outputd2 * dil
contourY = xpulse_Sd/pulse_int + Sd0_exp * dil

# set colorbar range to be the same

colormin = 0.0
colormax = round_up(np.max([fin_r1Den, fin_r2Den, fin_r3Den, fin_r4Den, fin_r5Den, fin_r6Den]), 1) 
colormax1 = round_up(np.max([fin_rHet, fin_rAOO, fin_rNOO, fin_rAOX]), 1) 

# plot
p1 = ax1.pcolormesh(contourX, contourY, fin_O2, cmap=colmap) #np.log(fin_O2+0.001)
p2 = ax2.pcolormesh(contourX, contourY, fin_NO3, cmap=colmap)
p3 = ax3.pcolormesh(contourX, contourY, fin_NO2, cmap=colmap) 
p4 = ax4.pcolormesh(contourX, contourY, fin_Sd, cmap=colmap)
p5 = ax5.pcolormesh(contourX, contourY, fin_N2O*1e3*0.5, cmap=colmap)

p6 = ax6.pcolormesh(contourX, contourY, fin_rHet, vmin=colormin, vmax=colormax1, cmap=colmap)
ax7.set_title('AMX contribution (%)', fontsize=fslab)
p7 = ax7.pcolormesh(contourX, contourY, fin_rAOX * nh4_n2_AOX / (fin_rAOX * nh4_n2_AOX + (fin_r2Den + fin_r3Den +fin_r5Den) * 0.5) * 100, cmap=colmap) 
## Add contour lines
masked_anammox_data = np.copy(fin_rAOX * nh4_n2_AOX / (fin_rAOX * nh4_n2_AOX + (fin_r2Den + fin_r3Den +fin_r5Den) * 0.5) * 100)
mask_boundary_value = 25  # set a value slightly lower than 30 to ensure only the edge is masked
mask = masked_anammox_data <= mask_boundary_value
# Apply the mask to the data by setting the undesired region to NaN
masked_anammox_data[mask] = np.nan
# Add contour lines 
contour_levels = [30]  # Define the levels at which you want contours
CS = ax7.contour(contourX, contourY, masked_anammox_data, levels=contour_levels, colors='black')
ax7.clabel(CS, inline=True, fontsize=8)  # Add labels to the contours

p8 = ax8.pcolormesh(contourX, contourY, fin_rAOO, vmin=colormin, vmax=colormax1, cmap=colmap)
p9 = ax9.pcolormesh(contourX, contourY, fin_rNOO, vmin=colormin, vmax=colormax1, cmap=colmap)

p10 = ax10.pcolormesh(contourX, contourY, fin_r1Den, vmin=colormin, vmax=colormax, cmap=colmap) 
p11 = ax11.pcolormesh(contourX, contourY, fin_r6Den, vmin=colormin, vmax=colormax, cmap=colmap)
p12 = ax12.pcolormesh(contourX, contourY, fin_r3Den, vmin=colormin, vmax=colormax, cmap=colmap) 
p13 = ax13.pcolormesh(contourX, contourY, fin_r4Den, cmap=colmap) #vmin=colormin, vmax=colormax, 
p14 = ax14.pcolormesh(contourX, contourY, fin_r2Den, vmin=colormin, vmax=colormax, cmap=colmap)
p15 = ax15.pcolormesh(contourX, contourY, fin_r5Den, vmin=colormin, vmax=colormax, cmap=colmap) 

## delete axis number of some subplots
ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic, labelbottom=False)
ax5.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax6.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax7.tick_params(labelsize=fstic, labelbottom=False)
ax8.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax9.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax10.tick_params(labelsize=fstic, labelbottom=False)
ax11.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax12.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax13.tick_params(labelsize=fstic)
ax14.tick_params(labelsize=fstic, labelleft=False)
ax15.tick_params(labelsize=fstic, labelleft=False)


## add axis title to some subplots
contourYlabel = 'OM supply with pulses (µM-N/d)'
contourXlabel = 'O$_2$ supply (µM/d)'

ax7.set_ylabel(contourYlabel, fontsize=fslab)


ax13.set_xlabel(contourXlabel, fontsize=fslab)
ax14.set_xlabel(contourXlabel, fontsize=fslab)
ax15.set_xlabel(contourXlabel, fontsize=fslab)

xlowerlimit = 0
xupperlimit = 10.02*dil
xtickdiv = 0.1
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)
cbar7 = fig.colorbar(p7, ax=ax7)
cbar8 = fig.colorbar(p8, ax=ax8)
cbar9 = fig.colorbar(p9, ax=ax9)
cbar10 = fig.colorbar(p10, ax=ax10)
cbar11 = fig.colorbar(p11, ax=ax11)
cbar12 = fig.colorbar(p12, ax=ax12)
cbar13 = fig.colorbar(p13, ax=ax13)
cbar14 = fig.colorbar(p14, ax=ax14)
cbar15 = fig.colorbar(p15, ax=ax15)

plt.tight_layout()
#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('FigS_allotherresults_with/withoutNOB.png', dpi=300) 

#%% FigS_Contour_copio-den-biomasses
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)


fig = plt.figure(figsize=(12,14))
gs = GridSpec(4, 3)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])

ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])
ax10 = plt.subplot(gs[3,0])
ax11 = plt.subplot(gs[3,1])
ax12 = plt.subplot(gs[3,2])


nh4_n2_AOX = (e_n2AOX*0.5*y_nh4AOX)
totalb = fin_b1Den + fin_b2Den + fin_b2Den + fin_b3Den + fin_b4Den + fin_b5Den + fin_b6Den\
    + fin_b1DenC + fin_b2DenC + fin_b3DenC + fin_b4DenC + fin_b5DenC + fin_b6DenC\
        + fin_bAOO + fin_bNOO + fin_bAOX + fin_bHet + fin_bHetC

def colbarmax(fin):
    return int(np.max(fin)*1.2)

contourX = outputd2 * dil
contourY = xpulse_Sd/pulse_int + Sd0_exp * dil

# set colorbar range to be the same
colormin = 0.0
colormax = 0.8

ax1.set_title('Biomass NO$_3$$^-$→NO$_2$$^-$ (µM-N)', fontsize=fslab)
p1 = ax1.pcolormesh(contourX, contourY, fin_b1Den, vmin=colormin, vmax=colormax, cmap=colmap) 

ax2.set_title('Biomass NO$_3$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p2 = ax2.pcolormesh(contourX, contourY, fin_b6Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax3.set_title('Biomass NO$_3$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p3 = ax3.pcolormesh(contourX, contourY, fin_b3Den, vmin=colormin, vmax=colormax, cmap=colmap) 

ax4.set_title('Biomass NO$_2$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p4 = ax4.pcolormesh(contourX, contourY, fin_b4Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax5.set_title('Biomass NO$_2$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p5 = ax5.pcolormesh(contourX, contourY, fin_b2Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax6.set_title('Biomass N$_2$O→N$_2$ (µM-N)', fontsize=fslab)
p6 = ax6.pcolormesh(contourX, contourY, fin_b5Den, vmin=colormin, vmax=colormax, cmap=colmap) 


ax7.set_title('Copio biomass NO$_3$$^-$→NO$_2$$^-$ (µM-N)', fontsize=fslab)
p7 = ax7.pcolormesh(contourX, contourY, fin_b1DenC, vmin=colormin, vmax=colormax, cmap=colmap)
ax8.set_title('Copio biomass NO$_3$$^-$→N$_2$O (µM-N)', fontsize=fslab) 
p8 = ax8.pcolormesh(contourX, contourY, fin_b6DenC, vmin=colormin, vmax=colormax, cmap=colmap)
ax9.set_title('Copio biomass NO$_3$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p9 = ax9.pcolormesh(contourX, contourY, fin_b3DenC, vmin=colormin, vmax=colormax, cmap=colmap) 
ax10.set_title('Copio biomass NO$_2$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p10 = ax10.pcolormesh(contourX, contourY, fin_b4DenC, vmin=colormin, vmax=colormax, cmap=colmap) 
ax11.set_title('Copio biomass NO$_2$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p11 = ax11.pcolormesh(contourX, contourY, fin_b2DenC, vmin=colormin, vmax=colormax, cmap=colmap) 
ax12.set_title('Copio biomass N$_2$O→N$_2$ (µM-N)', fontsize=fslab)
p12 = ax12.pcolormesh(contourX, contourY, fin_b5DenC, vmin=colormin, vmax=colormax, cmap=colmap) 


## delete axis title of some subplots
ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic, labelbottom=False) #
ax5.tick_params(labelsize=fstic, labelleft=False, labelbottom=False) 
ax6.tick_params(labelsize=fstic, labelleft=False, labelbottom=False) 
ax7.tick_params(labelsize=fstic, labelbottom=False)
ax8.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax9.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax10.tick_params(labelsize=fstic)
ax11.tick_params(labelsize=fstic, labelleft=False)
ax12.tick_params(labelsize=fstic, labelleft=False)

## add axis title to some subplots
contourYlabel = 'OM supply with pulses (µM-N/d)' 
contourXlabel = 'O$_2$ supply (µM/d)'


ax4.set_ylabel(contourYlabel, fontsize=fslab)

ax10.set_xlabel(contourXlabel, fontsize=fslab)
ax11.set_xlabel(contourXlabel, fontsize=fslab)
ax12.set_xlabel(contourXlabel, fontsize=fslab)

cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)
cbar7 = fig.colorbar(p7, ax=ax7)
cbar8 = fig.colorbar(p8, ax=ax8)
cbar9 = fig.colorbar(p9, ax=ax9)
cbar10 = fig.colorbar(p10, ax=ax10)
cbar11 = fig.colorbar(p11, ax=ax11)
cbar12 = fig.colorbar(p12, ax=ax12)


plt.tight_layout()


#%% Save the plot
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/figures")
fig.savefig('FigS_OligoCopio_biomass.png', dpi=300)


#%% Save the model output
#%% save the output to data folder
os.chdir("YourFolderPath/ChemostatModel_ModularDenitrification_clean/Output")

fname = 'OMpulse_'
np.savetxt(fname+'_pulse.txt', xpulse_Sd, delimiter='\t')
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
np.savetxt(fname+'_b7Den.txt', fin_b7Den, delimiter='\t')
np.savetxt(fname+'_bAOO.txt', fin_bAOO, delimiter='\t')
np.savetxt(fname+'_bNOO.txt', fin_bNOO, delimiter='\t')
np.savetxt(fname+'_bAOX.txt', fin_bAOX, delimiter='\t')

np.savetxt(fname+'_uHet.txt', fin_uHet, delimiter='\t')
np.savetxt(fname+'_u1Den.txt', fin_u1Den, delimiter='\t')
np.savetxt(fname+'_u2Den.txt', fin_u2Den, delimiter='\t')
np.savetxt(fname+'_u3Den.txt', fin_u3Den, delimiter='\t')
np.savetxt(fname+'_u4Den.txt', fin_u4Den, delimiter='\t')
np.savetxt(fname+'_u5Den.txt', fin_u5Den, delimiter='\t')
np.savetxt(fname+'_u6Den.txt', fin_u6Den, delimiter='\t')
np.savetxt(fname+'_u7Den.txt', fin_u7Den, delimiter='\t')
np.savetxt(fname+'_uAOO.txt', fin_uAOO, delimiter='\t')
np.savetxt(fname+'_uNOO.txt', fin_uNOO, delimiter='\t')
np.savetxt(fname+'_uAOX.txt', fin_uAOX, delimiter='\t')

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
