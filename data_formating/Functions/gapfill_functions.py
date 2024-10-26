# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:29:51 2022

@author: alexi
"""

import os
import pandas as pd

#%% Définition dictionnaires
station_dict = {'ARGENT': 'Roquemont',
                'AUXLOUPS': 'Lac Plétipi',
                'CABITUQG': 'Lac Cabituquimats',
                'CONRAD': 'Manouane A Météo', 
                'DIAMAND': 'Lac Cutaway',
                'GAREMANG': 'Garemand',
                # 'HARTJ',
                # 'LACROI',
                'LAFLAM': 'Laflamme',
                'LBARDO': 'Lac Bardoux',
                'LEVASSEU': 'Ile René Levasseur',
                # 'LOUIS',
                'LOUISE': 'Lac Louise',
                'MOUCHA': 'Mouchalagane',
                'NOIRS': 'Outardes-4 Sud',
                # 'PARENT',
                'PARLEUR': 'Lac Parleur',
                # 'PERDRIX',
                'PIPMUA': 'Réservoir Pipmuacan',
                'PORTO': 'Bersimis 1 Est',
                # 'ROUSSY',
                'RTOULNUS': 'Toulnustouc Nord-Est',
                'SAUTEREL': 'Rivières aux Sauterelles',
                # 'SM3CAM',
                'STMARG': 'Rivière Sainte-Marguerite rive Est',
                # 'WABISTAN',
                'WEYMOU': 'Weymont'}

hydromet_dict = {'tax000h': 'temp_max',
                 'tan000h': 'temp_min',
                 'tam000h': 'temp_moy',
                 'hai000h': 'humidite_air',
                 'nsi000h': 'neige_sol',}

#%% Gapfill hydromet
def hydromet_gapfill(data_path, hydromet_path, save_path):
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.set_index('date')
    
    hydromet = pd.read_csv(hydromet_path, sep=';', parse_dates=['date'])
    hydromet = hydromet.set_index('date')
    
    print('Début du gapfilling...\n')
    for station, subdf in df.groupby('filename'):
        hyd_subdf = (hydromet
                     .loc[hydromet['name'] == station_dict.get(station)])
        
        df_vars = list(hydromet_dict.values())
        hyd_vars = list(hydromet_dict.keys())
        
        for df_var, hyd_var in zip(df_vars, hyd_vars):
            gaps = subdf.loc[(subdf.index.isin(hyd_subdf.index)) &
                             (subdf[df_var].isna())].index
            
            if len(gaps) > 0:                
                subdf.loc[gaps, df_var] = hyd_subdf.loc[gaps, hyd_var]

        df.loc[(df['filename'] == station) &
               (df.index.isin(subdf.index))] = subdf
    
    print('Gapfilling terminé.\n')
    
    print('Sauvegarde dans %s' %save_path)
    df.to_csv(os.path.join(save_path, 'DATASET_GAPFILLED.csv'))
    
    return df

#%% Gapfill précipitation
def precip_gapfill():
    
    
    return