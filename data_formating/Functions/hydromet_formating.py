# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:48:48 2022

@author: alexi
"""

#%% Imports et path
import os
import glob
import pandas as pd
import numpy as np
import re

project_path =  os.path.abspath(os.path.join(__file__ ,"../.."))
data_path = os.path.join(project_path, 'Data', 'HQ', 'Hydrometeo', 'raw')
save_path = os.path.join(project_path, 'Data', 'HQ', 'Hydrometeo', 'stations')

import matplotlib.pyplot as plt
#%% Fonction principale
def hydromet_format_func(data_path, save_path):
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    file_names = [os.path.splitext(filename)[0] for filename in os.listdir(data_path)]
    
    var_dict = {"Horodatage (TUC)": "date", "FILENAME": "filename",
                "LUFFT_PRECIP_TOT_MM (11786)": "precip_tot_disdro",
                "LUFFT_PART_TYPE (11781)": "type_precip",
                "EEN_K (11335)": "EEN_K",
                "NEIGE_SOL_HORAIRE (801)": "neige_sol",
                "PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)": "precip_tot_pluvio",
                "TEMP_AIR_MAX_HORAIRE (204)": "temp_max",
                "TEMP_AIR_MIN_HORAIRE (205)": "temp_min",
                "TEMP_AIR_MOY_HORAIRE (203)": "temp_moy",
                 "DIR_VENT_MOY15MINUTES_15MINUTES_2,5METRES (405)": "dir_vent_moy_2_5m",
                "HUMIDITE_AIR_HORAIRE (301)": "humidite_air",
                "VIT_VENT_MOY15MINUTES_15MINUTES_2,5METRES (505)": "vitesse_vent_moy_2_5m",
                "DIR_VENT_MOY60MINUTES_HORAIRE_10METRES (403)": "dir_vent_moy_10m",
                "VIT_VENT_MOY60MINUTES_HORAIRE_10METRES (503)": "vitesse_vent_moy_10m",
                }
    
    
    for file, filename in zip(all_files, file_names):
        print(filename)
        station = re.split("Historique_|_2", filename)[1]
    
        print('Traitement des données hydrométéo à ' + station)
        df = pd.read_excel(file, parse_dates=["Horodatage (TUC)"])
        
        df.set_index("Horodatage (TUC)", inplace=True)
        df.rename(columns=var_dict, inplace=True)
        
        # Ajout de toutes les colonnes
        for var in var_dict.values():
            if var not in df.columns:
                df[var] = np.nan
        
    
        # Application des fonctions de formatage
        # TODO neige au sol
        for func in [precipitation, temperature, hum_rel]:
            df = func(df)
        
        # # # Ramène les colonnes au format initial
        for col in df.columns:
            if col in var_dict.values():    
                col2 = list(var_dict.keys())[list(var_dict.values()).index(col)]
                df.rename(columns={col: col2}, inplace=True)
                
        df.to_excel(os.path.join(save_path, (filename + '.xlsx')))
    return

#%% Fonctions
def precipitation(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    print('Traitement précipitation...\n')    
    
    var_list = ['precip_tot_pluvio']
    
    # Correction des mesures de précipitation cumulatives
    # Utilisation de diff pour détecter les "resets" de la somme de précipitation
    # soustraction de la valeur minimale de la sous-df pour ramener le tout à 0
    
    for col in var_list:
        # Ramène tous les points positifs
        df[col] += abs(df[col].min())
        # plt.plot(df.index,df[col])
        # Retrait des oscillations de la valeur cumulative
        diff = (df[col]
                .fillna(method='bfill')
                .diff()) # limit=12 trouvé par essai/erreur
        diff_mask = (diff < 0) | (diff >= (diff.mean() + 3*diff.std()))
        df.loc[diff_mask, col] = np.nan
        df[col].fillna(method='bfill')
        
        # Raccrocahge des données
        df.loc[df[col] < 0, col] = np.nan            
        diff = (df[col]
                .fillna(method='bfill', limit=12)
                .diff()) # limit=12 trouvé par essai/erreur
        diff_mask = ((diff < 0) | (diff > 50)).values
        col_index = (df
                     .columns
                     .get_loc(col))
        # plt.plot(df.index, df[col])
        # TODO généraliser la durée du pas de temps
        # pdt = (pd
        #        .to_timedelta(pd
        #                      .infer_freq(df
        #                                  .index)))
        pdt = pd.to_timedelta("00:15:00")
        
        window_end = df.iloc[diff_mask, col_index].index
        window_start = window_end - 10*pdt # testé pour 10*pdt, quantité à généraliser
        # TODO fin de l'opération à généraliser            
        
        for i in range(len(window_end)):
            # applique la correction seulement si window_start fait partie de l'index
            if window_start[i] > df.index.min():
                corr = (-diff
                        .loc[window_start[i]:window_end[i]]
                        .min())
                df.loc[window_end[i]:, col] += corr                            
    
        # Retrait des "jumps"            
        jumps = diff.copy()
        jumps.loc[(jumps < jumps.mean()+3*jumps.std()) |
                  (jumps < 50)] = 0 # 50 est une valeur arbitraire
        jumps = np.cumsum(jumps)
        df[col] = df[col] - jumps
        # plt.plot(df.index, df[col])
        # # Remise à zéro annuelle
        # year_list = (df
        #              .index
        #              .year
        #              .unique()
        #              .sort_values())
            
        # for year_start, year_end in zip(year_list[:-1], year_list[1:]):
        #     start = (str(year_start) + '-10-01')
        #     end = (str(year_end) + '-06-01')
        
        #     df.loc[((df.index >= start) & (df.index <= end)), 
        #            col] = df.loc[start: end, col] - df.loc[start: end, col].min()
        df[col] += -df[col].min()
        # plt.plot(df.index, df[col])
        # plt.show()
    return df

def temperature(df):

    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    print('Traitement température...\n')
    temp_var = ['temp_max', 'temp_min', 'temp_moy']    
    for var in temp_var:
        mu = df[var].mean()
        sig = 3*df[var].std()
        
        df.loc[(df[var] > mu + sig) | 
                  (df[var] < mu - sig), 
                  var] = np.nan
        df.loc[(df[var] > 40) | 
                  (df[var] < -50), 
                  var] = np.nan
        
        df[var].interpolate(limit=1)
        
    return df

def hum_rel(df):
    print('Traitement humidité relative...\n')
    hum_var = ['humidite_air']
    
    # TODO décider si on garde les valeurs > 100%
    for var in hum_var:
        df.loc[df[var] > 100, var] = np.nan
        df.loc[df[var] < 0, var] = np.nan
        
        diff = (df[var]
            .fillna(method='bfill', limit=12)
            .diff())
        
        mu = diff.mean()
        sig = 3*diff.std()
        
        df.loc[(diff > mu + sig) | 
                  (diff < mu - sig), 
                  var] = np.nan 
        
        df[var].interpolate(limit=1)
    
    return df
