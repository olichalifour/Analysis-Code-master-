# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:07:20 2022

@author: alexi
"""

import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


frac_list = ['f_0','f_60', 'f_67', 'f_69', 'f_70','f_nan']
frac_list_qty = ['#nan_1h']

#%% Fonctions pour passer du 15 min au 1h
def phase_frac(df_phase):


    phase_list = (df_phase['type_precip'].unique())
    df_phase[frac_list] = np.nan



    for phase in phase_list:

        if np.isnan(phase):
            df_phase.loc[df_phase['type_precip'].isna(),
                   f'#{phase}_1h'] = 1
            df_phase.loc[df_phase['type_precip'].isna(),
                         f'f_{phase}'] = df_phase.loc[df_phase['type_precip'].isna(),
                                       'precip_inst_pluvio']

        else:
            phase_col = 'f_' + str(int(phase))
            df_phase.loc[df_phase['type_precip'] == phase,
                   phase_col] = df_phase.loc[df_phase['type_precip'] == phase,
                                       'precip_inst_pluvio']


    resamp_phase = df_phase[frac_list + frac_list_qty].resample('1H', label='left').sum()

    resamp_phase['tot'] = resamp_phase[frac_list].sum(axis=1)

    # for frac in frac_list:
    #     resamp[frac] = resamp[frac].div(resamp['tot'])
    
    # resamp_phase.drop(columns=['tot'], inplace=True)


    resamp_phase.fillna(0, inplace=True)

    return resamp_phase

def calcul_wind_10m(df,z_1,z_2):
    z_0 = 0.01
    d = 0.4
    u_z1 = df['vitesse_vent_moy_2m']
    factor = np.log((z_2-d)/z_0)  / np.log((z_1-d)/z_0)
    u_z2 = u_z1 * factor
    return u_z2

def pdt_15min_to_1h(data_path):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print("Création de la base de données horaire...")
    all_files = os.listdir(data_path)   
    
    for file in all_files:
        if file.endswith('formated.csv'):

            df = pd.read_csv(os.path.join(data_path, file),
                             parse_dates=['date'])
    df = df.set_index('date')

    resample_1h_op = {
                      'EEN_K': 'last',
                      'humidite_air': 'mean' ,
                      'neige_sol': 'last',
                      'temp_moy': 'mean',
                      'temp_max': 'max',
                      'temp_min': 'min',
                      'precip_inst_pluvio': 'sum',
                      'precip_inst_disdro': 'sum',
                      'dir_vent_moy_2_5m': 'mean',
                      'vitesse_vent_moy_2_5m': 'mean',
                      'dir_vent_moy_10m': 'mean',
                      'vitesse_vent_moy_10m': 'mean',
                      'elevation': 'mean',
                      }
    
    print('Resampling...')
    df_1h = []
    for station, subdf in df.groupby('filename'):


        subdf.sort_index(inplace=True)

        # threshold geonor obs 1 mm / jour
        # subdf_1d = subdf.resample('1d', label='left').agg(resample_1h_op)
        # mask_threshold_prcp = (subdf_1d['precip_inst_pluvio']>1).resample('15T', label='left').ffill()
        #
        # subdf['precip_inst_pluvio'] = subdf['precip_inst_pluvio'].where(mask_threshold_prcp,0)

        # subdf.loc[subdf['precip_inst_pluvio'] < 0.2, ['precip_inst_pluvio']] = 0

        # threshold geonor


        resamp = phase_frac(subdf)

        subdf_1h = subdf.resample('1H', label='left').agg(resample_1h_op)

        subdf_1h.loc[subdf_1h['precip_inst_pluvio'] < 0.2, ['precip_inst_pluvio']] = 0
        subdf_1h.loc[subdf_1h['precip_inst_pluvio'] > 110, ['precip_inst_pluvio']] = np.nan

        # mask = subdf_1h_mask['precip_inst_pluvio'] > 0
        #
        # for frac in frac_list:
        #     # print(frac,np.sum(resamp.loc[mask, frac]))
        #     resamp.loc[mask, frac] = resamp.loc[mask, frac] / subdf_1h_mask.loc[mask, 'precip_inst_pluvio']
        #
        # resamp.drop(columns=['tot'], inplace=True)
        #
        # df_30min_stat = subdf.resample('0.5H', label='right').agg(resample_1h_op)
        #
        # df_30min_stat.loc[df_30min_stat['precip_inst_pluvio'] < 0.14, ['precip_inst_pluvio']] = 0
        #
        # subdf_1h_stat = df_30min_stat.resample('1H', label='right').agg(resample_1h_op).loc[:mask.index[-1]]
        #
        # for frac in frac_list:
        #
        #     resamp.loc[mask, frac] = resamp.loc[mask, frac] * subdf_1h_stat.loc[mask, 'precip_inst_pluvio']
        #     # print(frac, np.sum(resamp.loc[mask, frac]))

        subdf_1h_tot = pd.concat([subdf_1h,
                           resamp[frac_list + frac_list_qty]], axis=1)

        # Ajout des infos textuelles
        subdf_1h_tot['filename'] = station
        df_1h.append(subdf_1h_tot)


    
    df_1h = pd.concat(df_1h)

    print('Sauvegarde de la base de donnée horaire...')
    df_1h.to_csv(os.path.join(data_path, 'dataset_1h.csv'))
    return

#%% Fonctions pour passer du 1h au 3h

def pdt_1h_to_3h(data_path):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print("Création de la base de données horaire...")
    all_files = os.listdir(data_path)   
    
    for file in all_files:
        if file.endswith('1h.csv'):
            df = pd.read_csv(os.path.join(data_path, file),
                             parse_dates=['date'])
    df = df.set_index('date')
    
    resample_3h_op = {'EEN_K': 'last',
                      'humidite_air': 'mean' ,
                      'neige_sol': 'last',
                      'temp_moy': 'mean',
                      'temp_max': 'max',
                      'temp_min': 'min',
                      'precip_inst_pluvio': 'sum',
                      'precip_inst_disdro': 'sum',
                      'dir_vent_moy': 'mean',
                      'vitesse_vent_moy': 'mean',
                      }
    
    return


