# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:22:15 2021

Recueil des fonctions de formatage pour les données de:
    - précipitation
    - températures
    - équivalent en eau de la neige
    - hauteur de neige

@author: alexi
"""
import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta,date
import warnings
import matplotlib.pyplot as plt


#%% Fonction pour le pré-cleanup
# Fonction ne sera plus utilisée
def pre_cleanup(data_path, save_path):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    save_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    all_files = [fn for fn in glob.glob(os.path.join(data_path, '*.csv'))
         if not os.path.basename(fn).endswith(tuple(['cleaned.csv', 'formated.csv']))]

    df = []
    print("Compilation des données 15 min en un fichier...")
    for filename in all_files:        
        subdf = pd.read_csv(filename, parse_dates=['date'])                
        subdf = subdf.set_index('date')
        df.append(time_filter(subdf))
    
    df = pd.concat(df)
    df = df.sort_values(by=['filename'], kind='mergesort').sort_index()
        
    functions = [remove_negatives, precipitation]
    
    print('Nettoyage des données... \n')
    for func in functions:
        df = func(df)                  
    
    print('Sauvegarde de la base de données... \n')
    df.to_csv(os.path.join(save_path, 'dataset_15min.csv'))    
    return

#%% Fonction principale regroupant toutes les opérations de nettoyage des données
def format_func(data_path):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    save_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    # all_files = [fn for fn in glob.glob(os.path.join(data_path, '*.csv'))
    #      if not os.path.basename(fn).endswith(tuple(['cleaned.csv', 'formated.csv']))]
    all_files = [fn for fn in glob.glob(os.path.join(data_path, '*.csv'))
         if os.path.basename(fn).endswith(tuple(['15min.csv']))]
    
    df = []
    for filename in all_files:

        print('Traitement du fichier ' + filename + '... \n')
        df = pd.read_csv(filename, parse_dates=['date'])
        df = df.set_index('date')
        
        df = time_filter(df)

        functions = [data_error, remove_negatives,precipitation,
                     temperature , SWE, neige, hum_rel, vent]

        
        print('Nettoyage des données... \n')
        for func in functions:
            df = func(df)
        
        print('Sauvegarde de la base de données... \n')

        # df.to_csv((filename.split('.')[0] + '_formated.csv'))
        df.to_csv((filename.split('.')[0] +'.'+ filename.split('.')[1] + '_formated.csv'))


#%% time_filter
def time_filter(df):
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
    month_drop = [6, 7, 8, 9]
    df.drop(df.index[df.index.month.isin(month_drop)], inplace=True)
    
    return df
    

#%% remove_negatives
def remove_negatives(df):

    var_list = ['precip_tot_disdro', 'EEN_K',
                'neige_sol', 'precip_tot_pluvio',
                'humidite_air', 'dir_vent_moy_2_5m',
                'vitesse_vent_moy_2_5m', 'dir_vent_moy_10m',
                'vitesse_vent_moy_10m',]
    for var in var_list:
        df.loc[df[var] < 0, var] = np.nan
    return df
        
#%% precipitation
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

    # potentiellement retirer ça

    df.loc[~df['type_precip'].isin([0,60, 67, 69, 70]), 'type_precip'] = np.nan

    # df.loc[df['type_precip'].isna(), 'type_precip'] = 0



    var_types = ['precip_tot_pluvio', 'precip_tot_disdro']
    
    # Retrait des valeurs négatives
    for col in var_types:
        df.loc[df[col] < 0, col] = np.nan
    
    # Correction des mesures de précipitation cumulatives
    # Utilisation de diff pour détecter les "resets" de la somme de précipitation
    # soustraction de la valeur minimale de la sous-df pour ramener le tout à 0
    dict_precip = {'precip_tot_pluvio': 'precip_inst_pluvio',
                   'precip_tot_disdro': 'precip_inst_disdro'}

    fig = plt.figure(facecolor='white', figsize=(10, 6), dpi=150)
    spec = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(spec[0, 0])
    ax1 = fig.add_subplot(spec[0, 1])

    df_stat = {}
    list_raw = []
    list_date = []
    list_neg = []
    list_naf = []
    list_name = []
    for name, group in df.groupby('filename'):
        group = group.sort_index()

        g = group.loc['2020-10':'2022-06']
        list_raw.extend(g['precip_tot_pluvio'].values)
        list_name.extend([name]*len(g.index))
        list_date.extend(g.index.values)
        if name == 'PORTO':
            g = group.loc['2020-10':'2022-06']

            ax1.plot(g.index,g['precip_tot_pluvio'],label='Raw',c='tab:red')
        elif name == 'LEVASSEU':
            g = group.loc['2020-10':'2022-06']
            ax0.plot(g.index,g['precip_tot_pluvio'],label='Raw',c='tab:red')
        else:
            pass
        for col in var_types:

            inst = dict_precip.get(col)
            # Ramène tous les points positifs
            df[col] += abs(df[col].min())
            
            # Retrait des oscillations de la valeur cumulative
            diff = (group[col]
                    .fillna(method='bfill')
                    .diff()) 
            diff_mask = (diff < 0) | (diff >= (diff.mean() + 3*diff.std()))
            group.loc[diff_mask, col] = np.nan
            group[col].fillna(method='bfill', inplace=True)


            # Raccrochage des données 
            # diff = (group[col]
            #     .fillna(method='bfill', limit=12)
            #     .diff())
            # diff_mask = (diff < 0).values
            # col_index = (group
            #              .columns
            #              .get_loc(col))
            # # TODO généraliser la durée du pas de temps
            # # pdt = (pd
            # #        .to_timedelta(pd
            # #                      .infer_freq(group
            # #                                  .index)))
            # pdt = pd.to_timedelta("00:15:00")
            #
            # window_end = group.iloc[diff_mask, col_index].index
            # window_start = window_end - 10*pdt # testé pour 10*pdt, quantité à généraliser
            # # TODO fin de l'opération à généraliser
            #
            # for i in range(len(window_end)):
            #     # applique la correction seulement si window_start fait parti de l'index
            #     if window_start[i] > group.index.min():
            #         corr = (-diff
            #                 .loc[window_start[i]:window_end[i]]
            #                 .min())
            #         group.loc[window_end[i]:, col] += corr
            group[inst] = group[col].diff()

            group.loc[(group[inst] < 0) | (group[inst] > 75), inst] = 0
            group[col] = group[inst].cumsum()

            diff = (group[col]
                    .fillna(method='bfill', limit=12)
                    .diff())
            # Retrait des "jumps"            
            jumps = diff.copy()
            jumps.loc[(jumps < jumps.mean() + 3*jumps.std()) |
                      (jumps < 50)] = 0
            jumps = np.cumsum(jumps)
            group[col] = group[col] - jumps


            # Remise à zéro annuelle
            year_list = (df
                          .index
                          .year
                          .unique()
                          .sort_values())
                
            for year_start, year_end in zip(year_list[:-1], year_list[1:]):
                start = (str(year_start) + '-10-01')
                end = (str(year_end) + '-06-01')
                
                df.loc[(df['filename'] == name) &
                        ((df.index >= start) &
                        (df.index <= end)),
                        col] = group.loc[start: end,
                                        col] - group.loc[start: end,
                                                          col].min()
        g = group.loc['2020-10':'2022-06']
        list_neg.extend(g['precip_tot_pluvio'].values)
        if name == 'PORTO':
            g = group.loc['2020-10':'2022-06']
            ax1.plot(g.index,g['precip_tot_pluvio'],label='Treated',c='tab:blue')
        elif name == 'LEVASSEU':
            g = group.loc['2020-10':'2022-06']
            ax0.plot(g.index,g['precip_tot_pluvio'],label='Treated',c='tab:blue')
        else:
            pass


    # Passage de précipitation cumulative à instantanée
    df['precip_inst_disdro'] = np.nan
    df['precip_inst_pluvio'] = np.nan

    precip_list = ['precip_inst_pluvio', 'precip_inst_disdro']
    dict_precip = {'precip_tot_pluvio': 'precip_inst_pluvio',
                    'precip_tot_disdro': 'precip_inst_disdro'}
    for name, group in df.groupby('filename'):

        for cumul in dict_precip.keys():
            inst = dict_precip.get(cumul)
            df.loc[df['filename'] == name, inst] = group[cumul].diff()

            # Retrait de précip instantanées négatives ou aberrantes
            df.loc[(df['filename'] == name) &
                   ((df[inst] < 0)),inst] = 0
    #


    
    # Retrait des données aberrantes de précip instantanée en début d'année
    for name, group in df.groupby('filename'):
        for year in df.index.year.unique()[:-1]:
            start = (str(year) + '-10-01')
            idx = group.loc[group.index >= start,
                        'precip_inst_pluvio'].first_valid_index()

            df.loc[(df.index == idx) & (df['filename'] == name),
                    'precip_inst_pluvio'] = 0



    # Réaccumulation complète des precip inst pour pouvoir appliquer NAF_SEG
    for name, group in df.groupby('filename'):
        group.sort_index(inplace=True)
        # filter
        # print(np.sum(group.index.duplicated()))

        for cumul in dict_precip.keys():        
            inst = dict_precip.get(cumul)
            # plt.plot(group.index, group[inst].cumsum())
            # df.loc[df['filename'] == name, cumul] =  group[inst].cumsum()

            if sum(group[cumul].isna()) < len(group[cumul]):
                xt = df.index
                xRawCumPcp = group[inst].cumsum() # Réaccumulation
                intPcpTh = 0.001
                nRecsPerDay = 24*4
                nWindowsPerDay = 3
                output_type = 'dataframe'
                
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore',
                                          category=RuntimeWarning)
                    out_NAF = NAF_SEG(xt, xRawCumPcp, intPcpTh, nRecsPerDay,
                                      nWindowsPerDay, output_type)
                
                df.loc[df['filename'] == name, cumul] =  out_NAF['cumPcpFilt']
                df.loc[df['filename'] == name, inst] =  out_NAF['cumPcpFilt'].diff()

        g = group.loc['2020-10':'2022-06']
        list_naf.extend(g['precip_inst_pluvio'].values)

        if name == 'PORTO':
            g = group.loc['2020-10':'2022-06']
            ax1.plot(g.index,np.cumsum(g['precip_inst_pluvio']),label='NAF-S',c='tab:green')
        elif name == 'LEVASSEU':
            g = group.loc['2020-10':'2022-06']
            ax0.plot(g.index,np.cumsum(g['precip_inst_pluvio']),label='NAF-S',c='tab:green')
        else:
            pass

    ax0.set_title('LEVASSEU')
    ax0.set_xlabel('Date [-]')
    ax0.set_ylabel('Accumulation [mm]')
    ax0.set_xlim([date(2020, 10, 1), date(2022, 6, 1)])
    ax1.set_title('PORTO')
    ax1.set_xlim([date(2020, 10, 1), date(2022, 6, 1)])
    ax1.legend()
    plt.show()
    dict = {'date':list_date,'RAW':list_raw,'neg':list_neg,'NAF':list_naf,'filename':list_name}
    df_acc = pd.DataFrame(dict,index = list_date,columns=['date','RAW','neg','NAF','filename'])
    df_acc.to_csv('/Users/olivier1/Documents/GitHub/data_format-master/Data.nosync/df_accumulation.csv')

    print('Traitement précipitation terminé.\n')
    return df



elev_dict_stat_meteo={
                'ROQUEMON':641,
                'PLETIPI': 537.9,
                'BAUBERT': 541,
                'BETSIA_M': 403, # à valider
                'LCABITUQ':491.0,
                'CONRAD': 433,
                'MANOUA_M': 537,
                'lCUTAWAY':440.0,
                'GAREMAND': 762,
                'HART_JAU': 460,
                'LACCROIX': 621,
                'LAFLAMME': 519,
                'LAVAL': np.nan,
                'LBARDOUX': 486,
                'RLEVASSE': 466,
                'LLOUISE': 420,
                'MIQUELON': 315,
                'MOUCHA_M': 565,
                'OUT_4_S1': 357.8,
                'PARENT_G': 442,
                'PARLEUR': 485,
                'LSTEANN2': np.nan,
                'PIPMUACA': 566.2,
                'BERS_1E1': 406,
                'ROMA_SE': np.nan,
                'LROUSSY': 456,
                'RTOULNUS': 688,
                'STMARGC': 461,
                'WAGEGUMA': 565,
                'WEYMOUNT': 363,
                'NA':np.nan,}

elev_dict = {   'ARGENT': 641,
                'AUXLOUPS': 537.9,
                'BAUBERT': 541,
                'BETSIA_M': 403, # à valider
                'CABITUQG': 491,
                'LCABITUQ':491.0,
                'CONRAD': 433,
                'DIAMAND': 373,
                'GAREMANG': 778, # à valider
                'HARTJ_G': 460,
                'LACROI_G': 621,
                'LAFLAM_G': 519,
                'LAVAL': np.nan,
                'LBARDO_G': 486,
                'LEVASSEU': 466,
                'LOUISE_G': 397,
                'LOUIS': 315,
                'MOUCHA_M': 565,
                'NOIRS':385,
                'PARENT_G': 442,
                'PARLEUR': 485,
                'PERDRIX': 315,
                'PIPMUA_G': 566.2,
                'PORTO': 413,
                'ROMA_SE': np.nan,
                'ROUSSY_G': 456,
                'RTOULNUS': 688,
                'SM3CAM_G': 522,
                'STMARG_G': 461,
                'SAUTEREL':459 ,
                'WABISTAN': 565,
                'WEYMOU_G': 400,
                'NA':np.nan,
                    }
st_dict = {'ARGENT': 'ROQUEMON',
               'AUXLOUPS': 'PLETIPI',
               'BAUBERT': 'BAUBERT', # hr déjà dispo
               'BETSIA_M': 'BETSIA_M', # hr déjà dispo
            'CABITUQG': 'LCABITUQ', # gapfill manquant
                'CONRAD': 'MANOUA_M', # gapfill manquant
                'DIAMAND': 'lCUTAWAY', # gapfill manquant
                'GAREMANG': 'GAREMAND',
                'HARTJ_G': 'HART_JAU',
                'LAVAL': 'NA',
                'LACROI_G': 'LACCROIX',
                'LAFLAM_G': 'LAFLAMME',
                'LBARDO_G': 'LBARDOUX',
                'LEVASSEU': 'RLEVASSE',
                'LOUISE_G': 'LLOUISE',
                'LOUIS': 'MIQUELON',
                'MOUCHA_M': 'MOUCHA_M', # hr déja dispo
                'NOIRS': 'OUT_4_S1',
            'PARENT_G': 'PARENT_G', # gapfill manquant
                'PARLEUR': 'NA', # hr déjà dispo
                'PERDRIX': 'LSTEANN2',
                'PIPMUA_G': 'PIPMUACA',
                'PORTO': 'BERS_1E1',
            'ROMA_SE': 'NA', # gapfill manquant
                'ROUSSY_G': 'LROUSSY',
                'RTOULNUS': 'NA', # hr déjà dispo
                'SAUTEREL': 'NA', # hr déjà dispo
            'SM3CAM_G': 'NA', # gapfill manquant
                'STMARG_G': 'STMARGC', # hr déjà dispo
                'WABISTAN': 'WAGEGUMA',
                'WEYMOU_G': 'WEYMOUNT'
                }
#%% temperature
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
    temp_var = ['temp_max', 'temp_min',
                'temp_moy']    
    for station, subdf in df.groupby('filename'):
        for var in temp_var:
            mu = subdf[var].mean()
            sig = 3*subdf[var].std()
            
            subdf.loc[(subdf[var] > mu + sig) | 
                      (subdf[var] < mu - sig), 
                      var] = np.nan
            subdf.loc[(subdf[var] > 40) | 
                      (subdf[var] < -50), 
                      var] = np.nan
            
            diff = subdf[var].diff()
            mu = diff.mean()
            sig = 3*diff.std()            
            subdf.loc[(diff > mu + sig) |
                      (diff < mu - sig),
                      var] = np.nan
            
            subdf[var].interpolate(method='linear', limit=9, inplace=True)

            df.loc[df['filename'] == station, var] = subdf[var]
    print('Traitement température terminé.\n')

    return df

#%% hum_rel
def hum_rel(df):
    print('Traitement humidité relative...\n')
    hum_var = ['humidite_air']
    
    for station, subdf in df.groupby('filename'):
        for var in hum_var:
            subdf.loc[subdf[var] > 100, var] = np.nan
            # On juge les valeurs sous 10% comme étant erronées
            subdf.loc[subdf[var] < 10, var] = np.nan
            diff = (subdf[var]
                .fillna(method='bfill', limit=12)
                .diff())
            
            mu = diff.mean()
            sig = 3*diff.std()
            
            subdf.loc[(diff > mu + sig) | 
                      (diff < mu - sig), 
                      var] = np.nan
            
            subdf[var].interpolate(method='linear', limit=5, inplace=True)
            df.loc[df['filename'] == station, var] = subdf[var]
    
    print('Traitement humidité relative terminé.\n')
    return df

#%% vent
def vent(df):
    vent_var = ['dir_vent_moy_2_5m', 'vitesse_vent_moy_2_5m', 'dir_vent_moy_10m', 'vitesse_vent_moy_10m']

    for station, subdf in df.groupby('filename'):
        for var in vent_var:
            if var == 'dir_vent_moy_10m' or var == 'dir_vent_moy_2_5m':
                subdf.loc[subdf[var] > 360, var] = np.nan
            elif var == 'vitesse_vent_moy_2_5m' or var == 'vitesse_vent_moy_10m':
                subdf.loc[subdf[var] > 150, var] = np.nan
                subdf.loc[subdf[var] < 0, var] = np.nan
            # subdf[var].interpolate(method='linear', limit=4, inplace=True)
            df.loc[df['filename'] == station, var] = subdf[var]
    
    return df

#%% SWE
def SWE(df):
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
    print('Traitement SWE...\n')
    snow_var = ['EEN_K']
    station_groups = df.groupby('filename')    
    
    for station, subdf in station_groups:        
        for var in snow_var:
            # Retrait des valeurs <0
            mask = (subdf[var] < 0)
            subdf.loc[mask, var] = np.nan    
            
            # Retrait des valeurs aberrantes selon la distribution des données
            outliers = subdf[var].mean() + 3*subdf[var].std()
            subdf.loc[(subdf[var].notna()) &
                      (subdf[var] > outliers), 
                      var] = np.nan
                  
            # Retrait des pics et creux aberrants en fixant un seuil selon 
            # distribution de la variation des données
            diff = subdf.loc[subdf[var].notna(), var].diff()    
            threshold = abs(diff.mean()) + 3*diff.std()
            
            subdf.loc[(subdf[var].notna()) &
                      (abs(diff) > threshold), 
                      var] = np.nan
                        
            # Interpolation et fill des données
            # limit=16 trouvé par essai/erreur
            subdf[var].interpolate(method='linear', limit=16, inplace=True)
            subdf[var].fillna(method='ffill', limit=16, inplace=True) 
            subdf.loc[subdf[var] == 0, var] = np.nan
            
            # Retrait des "jumps"
            # TODO modifier pour ce use case, ils sont difficiles à différencier des vrais jumps
            # jumps = diff.copy()
            # jumps.loc[jumps < 50] = 0
            # jumps = np.cumsum(jumps)
            # subdf[var] = subdf[var] - jumps
            
        df.loc[df['filename'] == station, snow_var] = subdf[var]
    print('Traitement SWE terminé.\n')
    return df
                        
#%% Neige
def neige(df):
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
    print('Traitement neige...\n')
    station_groups = df.groupby('filename')
    snow_var = ['neige_sol']

    for station, subdf in station_groups:
        for var in snow_var:
            # Retrait des valeurs <0
            mask = (subdf[var] < 0)
            subdf.loc[mask, var] = np.nan    
            
            # Retrait des valeurs aberrantes selon la distribution des données
            outliers = subdf[var].mean() + 3*subdf[var].std()
            subdf.loc[(subdf[var].notna()) &
                      (subdf[var] > outliers), 
                      var] = np.nan
                  
            # Retrait des pics et creux aberrants en fixant un seuil selon 
            # distribution de la variation des données
            diff = subdf.loc[subdf[var].notna(), var].diff()    
            threshold = abs(diff.mean()) + 3*diff.std()
            
            subdf.loc[(subdf[var].notna()) &
                      (abs(diff) > threshold), 
                      var] = np.nan
                        
            # Interpolation et fill des données
            # limit=16 trouvé par essai/erreur
            subdf[var].interpolate(method='linear', limit=16)
            subdf[var] = subdf[var].fillna(method='ffill', limit=16) 
            subdf.loc[subdf[var] == 0, var] = np.nan
            
        df.loc[df['filename'] == station, snow_var] = subdf[var]
    print('Traitement neige terminé.\n')
    return df

#%% data_error
def data_error(df):
    # Variables à retirer
    station_list = ['PIPMUA_G',
                    'PARENT_G',
                    'BETSIA_M',
                    'MOUCHA_M',
                    ]
    
    variables = [['temp_moy', 'temp_max', 'temp_min'], 
                 ['EEN_K'],
                 ['precip_tot_pluvio'],
                 ['precip_tot_pluvio'],
                 ['precip_tot_pluvio'],
                 ]
    
    start_list = ['2019-10-01',
                  '2021-03-26',
                  '2019-10-01',
                  '2021-03-15',
                  ]
    
    end_list = ['2020-10-01',
                '2021-10-01',
                '2021-06-01',
                '2021-06-01',
                ]
    
    for station, var, start, end in zip(station_list, variables, start_list, end_list):
        df.loc[(df['filename'] == station) & 
               (df.index >= start) &
               (df.index < end),
               var] = np.nan        
    
    return df

#%% df_load
def df_load(path_list, df_savepath):
    
    dataframe = []
    for path in path_list:
        print('Ajout des bases de données dans ' + path + '\n')
        all_files = glob.glob(os.path.join(path, '*_formated.csv'))

        for filename in all_files:
            df = pd.read_csv(filename, parse_dates=['date'])
            dataframe.append(df)
            
    dataframe = pd.concat(dataframe, axis=0)
    dataframe = dataframe.set_index('date')
    
    # Assignation de types de données appropriées
    str_type = ['filename', 'source', 'disdrometer_type', 'pluviometer_type']
    
    dtype_dict = {}
    keys = list(dataframe.columns)
    for k in keys:
        if k in str_type:
            dtype_dict[k] = 'string'
        else:
            dtype_dict[k] = 'float64'
            
    dataframe.astype(dtype_dict)    
    
    print('Sauvegarde de la base de données complète dans')
    print(df_savepath + '\n')
    dataframe.to_csv(os.path.join(df_savepath, 'FULL_DATASET.csv'))
    return dataframe



#%% NAF filter
# Fonctions nécessaires pour faire rouler NAF_SEG sur les observations de précipitation
def NAF_SEG(xt,xRawCumPcp,intPcpTh,nRecsPerDay,nWindowsPerDay, output_type='dataframe'):
    '''
    VERSION 1.2 20200420
    PYTHON version 1.0
    ***********************************************************************
    Written by A. Barr & Amber Ross, Environment and Climate Change Canada,
    Climate Research Division, Saskatoon, Canada, 8 Dec 2018
    Transfert into Python by Jean-Benoit Madore Universite de Sherbrooke
    Sherbrooke, Quebec, Canada, April 2022
    ***********************************************************************
    
    The Segmented Neutral Aggregating Filter (NAF_SEG) 
    filters a cumulative precipitation time series xRawCumPcp
    24 hours at a time (within overlapping moving windows)
    using the brute-force NAF filter (Smith et al., 2019)
    and produces an estimate of evaporation on days without 
    precipitation.
    
    Syntax:
    df_NAF_SEG = NAF_SEG ...
        (xt,xRawCumPcp,intPcpTh,nRecsPerDay,nWindowsPerDay)
    
    Inputs:
    
    xt: datetime array : datetime format dates associated with the time series
    
    xRawCumPcp: numpy array: raw cumulative precipitation time series
    
    intPcpTh: float: desired minimum interval precipitation P*
    (intPcpTh=0.001 is recommended)
    
    nRecsPerDay: int: number of measurements per day
                     example: hourly data -> nRecsPerDay=24
                     
    nWindowsPerDay: int: number of overlapping moving windows per day - 
           must be a factor of nRecsPerDay (nWindowsPerDay=3 is recommended)
    output_type: string: type of output variable:
        -> 'dataframe' (default): panda dataframe
        -> 'dictionary': python dictionary
           
    Outputs:
    out_NAF_SEG: pandas datafarame/dictionary with columns/keys:
       -> t: as xt but with complete days of nRecsPerDay. Returned
          t will be in datenum format
       -> cumPcpFilt: filtered cumulative precipitation time series vector
       -> cumEvap: inferred cumulative evaporation time series vector
    
    The filtering is done using a brute-force algorithm NAF
    that identifies small or negative changes (below intPcpTh)
    then transfers them to neighbouring positive changes
    thus aggregating all changes to values above intPcpTh.
    
    Note that the precipitation on the first and last days of the time series
    are adversely impacted unless the bucket weight time series is "padded"
    with 1 day of fictitious precipitation at the beginning and end of the
    time series. To remedy this, add 1 full day of zeros to the beginning of
    the data and one full day of max bucket weight values to the end of the
    data. This will allow the algorithm to make a precipitation estimate for
    those two days.
    
    Revisions:
    4 Oct 2019 (Craig Smith) Replaced the call to the function PcpFiltPosTh
                with the call to NAF which is the version published in Smith
                et al. (2019).
                Fixed the code so that t can be passed to the function in
                either datenum or datetime format. datetime is converted to
                datenum
    20 Apr 2020 (Amber Ross) Removed un-used mArrays and sArrays structures  
                from the function exports. Cleaned up code and comments. 
    15 Apr 2022 (Jean-Benoit Madore) Adapted for python

    --------------------------------------------------------------------------
    References:
    Smith, C. D., Yang, D., Ross, A., and Barr, A.: The Environment and 
    Climate Change Canada solid precipitation intercomparison data from 
    Bratt's Lake and Caribou Creek, Saskatchewan, Earth Syst. Sci. Data, 11,
    1337–1347, https://doi.org/10.5194/essd-11-1337-2019, 2019. 
    --------------------------------------------------------------------------

    '''

    fDetect = 'All' # Should be default. JB: havent explore this parameter
    # fDetect = 'Any' # Test des autres options 'Any' ou 'Half' # TODO vérifier l'effet

    nRecsPerBlock = nRecsPerDay/nWindowsPerDay # moving window increment


    xt = pd.DatetimeIndex(pd.to_datetime(xt))# work in pandas date format

    tDay = np.unique(xt.date) # Make sure there is no duplicates

    dt = 1/nRecsPerDay # Fraction of day to use with timedelta

    # Setting theorical timeseries
    t = pd.date_range(start = tDay[0]+timedelta(days=dt), end = tDay[-1], freq=str(dt)+'D')
    # lenght of timeserie
    nt = len(t)
    # create an empty series to put the cumulative precip
    cumPcpRaw = pd.Series(np.nan, index=t)
    
    # create an empty series to put the intensity precip
    intPcpRaw = pd.Series(np.nan, index=t)
    
    # map xt onto t
    ftMap = xt.intersection(t)

    # Put the raw precip in the series. Any missing time will be np.nan

    cumPcpRaw[ftMap] = xRawCumPcp
    
    # Find all non np.nan values
    itYaN = np.where(~np.isnan(cumPcpRaw.values))[0]
    
    # Discrete diffenrence of all records to deacumulate
    # TODO ligne initiale en comment
    # intPcpRaw[itYaN[1:]] = cumPcpRaw[itYaN].diff()
    intPcpRaw[itYaN] = cumPcpRaw[itYaN].diff()
    
    # Fill first value with 0. The value was discarted during the deacumulation
    # TODO modifier pour le cas d'une station sans observations - sûrement en amont de la fonction anyway
    intPcpRaw[itYaN[0]] = 0

    # Create np.nan matrices to fill with filtered data
    intPcpArray=np.empty([nt,nWindowsPerDay]); # interval precipitation
    intPcpArray[:] = np.nan
    intEvapArray=np.empty([nt,nWindowsPerDay]) # interval evaporation
    intEvapArray[:] = np.nan
    flagPcpArray=np.empty([nt,nWindowsPerDay]); # interval precipitation flag
    flagPcpArray[:] = np.nan
    flagEvapArray=np.empty([nt,nWindowsPerDay]); # interval evaporation flag
    flagEvapArray[:] = np.nan


    # Iterate index from 0 to lenght of the time series minus the last day's measurements
    for itW1 in np.arange(0.0, nt+1-nRecsPerDay, nRecsPerBlock): 

        # index of the 24h later measurment
        itW2 = itW1 + nRecsPerDay # All measurement of the day
        
        # indexes that are evaluated
        itW = np.arange(itW1,itW2).astype(int)

        # Find data in cumPcpRaw within itW indexes
        jtYaN = np.where(~np.isnan(cumPcpRaw[itW]))[0]

        # number of valid values within the evaluated day
        ntYaN=len(jtYaN)

        # Find np.nan in cumPcpRaw within itW index   
        jtNaN = np.where(np.isnan(cumPcpRaw[itW]))[0]   
        
        # TODO autres cas??? Dans Ross 2020, ntYaN >= 2 n'est pas un critère de filtration
        # ce IF vient du code de l'auteur?
        if ntYaN>=2: # Case where we can apply the filter if we have more than 2 data points

            #if 24h accumulation dPcpW >= intPcpTh, filter and treat as Precip
            #% if 24h accumulation dPcpW <= -intPcpTh, filter and treat as Evap
            #% otherwise set both Pcp and Evap to zero.

            # Cumulative precipiation for the 24h evaluated
            dPcpW=cumPcpRaw[itW[jtYaN[-1]]] - cumPcpRaw[itW[jtYaN[0]]]

            ## Check if the cumulative precipitation over the evaluated 24h is:
            # more then the minimal threshold (intPcpTh), : precip
            # bellow intPcpTh but above negative intPcpTh ( -intPcpTh < dPcpW > intPcpTh) : no precip
            # else: the cumulative is below the negative intPcpTh: evaporation

            # Case the precipitation is greater than the minimal threshold
            if dPcpW>=intPcpTh: 

                # Precip detected within the window. Filter the 24h with NAF
                tmpCumPcpFilt=NAF(cumPcpRaw[itW],intPcpTh)  


                # create nan matrix of nb of rec per day
                tmpIntPcpFilt=np.empty(nRecsPerDay) 
                tmpIntPcpFilt[:] = np.nan

                #deaccumulate precip and fill it in the new matrix
                tmpIntPcpFilt[jtYaN[1:]] = np.diff(tmpCumPcpFilt[jtYaN])
                tmpIntPcpFilt[jtYaN[0]] = np.nan

                # create an array of zeros of size nRecsPerDay
                tmpIntEvap = np.zeros(nRecsPerDay) 


                # fill the nan values identify by jtNaN
                tmpIntEvap[jtNaN] = np.nan 


                # We consider that all precip are there. no evap
                # Create an array of ones of size nRecsPerDay
                flagPcp = np.ones(nRecsPerDay) 

                #flagEvap=zeros(nRecsPerDay,1);
                # no evap 
                flagEvap = np.zeros(nRecsPerDay) # Create an array of zeros of size nRecsPerDay

            # Case the cumul precipitation lower than intPcpTh but greater than negative intPcpTh (-intPcpTh < dPcpW > intPcpTh) 
            elif dPcpW>-intPcpTh: #% assumed to be zero

                #There is no precip or evaporation. everything flag to 0
                tmpIntPcpFilt = np.zeros(nRecsPerDay)
                tmpIntPcpFilt[jtNaN] = np.nan

                #There is no precip or evaporation. everything flag to 0
                tmpIntEvap = np.zeros(nRecsPerDay)
                tmpIntEvap[jtNaN] = np.nan

                #There is no precip or evaporation. everything flag to 0
                flagPcp=np.zeros(nRecsPerDay)

                #There is no precip or evaporation. everything flag to 0
                flagEvap = np.zeros(nRecsPerDay)

            # dPcpW < -intPcpTh -- > evaporation
            else: #% evap

                # No precipitation where recorded set precip to 0
                tmpIntPcpFilt=np.zeros(nRecsPerDay)
                tmpIntPcpFilt[jtNaN] = np.nan

                # pass NAF to the opposed evaporation. then reverse it to fit the evap
                tmpCumEvap=-NAF(-cumPcpRaw[itW],intPcpTh)

                #create an empty array to be fill by evap
                tmpIntEvap=np.empty(nRecsPerDay)
                tmpIntEvap[:] = np.nan

                #Fill with the evap value 
                tmpIntEvap[jtYaN[1:]] = np.diff(tmpCumEvap[jtYaN])
                tmpIntEvap[jtYaN[0]] = np.nan

                #No precip
                flagPcp = np.zeros(nRecsPerDay)

                #All evap
                flagEvap = np.ones(nRecsPerDay)

            # Evaluate on the different windows 
            for iW in range(0, nWindowsPerDay):
              # itArray is the iteration array which will change and move to correspond
              # to the evaluated time.

                #Case of the firt data of the dataset
                # itArray is the same as itW

                if itW1 == 0:
                    itArray=itW 
                    jtFilt=itW 

                # Case of the last evaluated day of the dataset
                elif itW1 ==  nt-nRecsPerDay:

                    it1 = itW1+ (iW*nRecsPerBlock)

                    it2 = itW1+nWindowsPerDay*nRecsPerBlock

                    itArray=np.arange(it1,it2).astype(int)

                    jtFilt = (itArray-it1).astype(int)

                # All other evaluations
                else:
                    it1=itW1+(iW*nRecsPerBlock) # Displace the index corresponding start to the window
                    it2=itW1+(iW+1)*nRecsPerBlock # Displace the index corresponding end to the window

                    itArray=np.arange(it1,it2).astype(int) # create list of int containing index of evaluated data
                    jtFilt= (itArray-itW1).astype(int) # Make it correspond to the index of the temporary variable

                ## Output from the temporary variable are put in the main variable
                intPcpArray[itArray, iW] = tmpIntPcpFilt[jtFilt]
                intEvapArray[itArray,iW] = tmpIntEvap[jtFilt]
                flagPcpArray[itArray,iW] = flagPcp[jtFilt]
                flagEvapArray[itArray,iW] = flagEvap[jtFilt]



    # Fill regular gaps in Window 1 using Window 2,
    # Window 1 always has extra missing values because
    # the first interval of each 24-h period is always missing.
        
    if nWindowsPerDay>1:
        #itW1gf=find(isnan(intPcpArray(:,1)) & ~isnan(intPcpArray(:,2)));
        # Find the non np.nan precip value of the Window 2 (~np.isnan(intPcpArray[:,1]) and np.nan 
        # precip values of window 1 (np.isnan(intPcpArray[:,0])
        itW1gf = np.where((np.isnan(intPcpArray[:, 0])) &
                          (~np.isnan(intPcpArray[:, 1])))[0]

        # Fill precip for window 1
        intPcpArray[itW1gf, 0] = intPcpArray[itW1gf, 1]

        # Find the non np.nan evap value of the Window 2 (~np.isnan(intPcpArray[:,1]) and np.nan 
        #   evap values of window 1 (np.isnan(intPcpArray[:,0])
        itW1gf = np.where(np.isnan(intEvapArray[:, 0]) &
                          (~np.isnan(intEvapArray[:, 1])))

        # Fill evap for window 1
        intEvapArray[itW1gf,0] = intEvapArray[itW1gf,1]

    # There is just 1 window per day. Fill the gap with 0 using gapsize()
    else:
        #ftGF=gapsize(intPcpArray)==1;
        ftGF=gapsize(intPcpArray) == 1
        #intPcpArray(ftGF)=0;
        intPcpArray[ftGF] = 0

        #ftGF=gapsize(intEvapArray)==1;
        ftGF=gapsize(intEvapArray) == 1
        intEvapArray[ftGF] = 0;

    # Use RestoreGaps to fill missing data. See the function for details
    intPcpArray,flagPcpArray = RestoreGaps(intPcpArray,flagPcpArray,nWindowsPerDay,cumPcpRaw,nt)
    
    # Reacumulate precip
    cumPcpArray=np.nancumsum(intPcpArray, axis=0)
    
    # Reacumulate evap
    cumEvapArray=np.nancumsum(intEvapArray, axis=0)     

    # Average every timestep over all windows for precip
    intPcpFilt = np.nanmean(intPcpArray,axis=1) 
    
    # Sum every timestep over all windows for precip flag
    nFlagPcp = np.nansum(flagPcpArray,axis=1)

    # Average every timestep over all windows for evap
    intEvap = np.nanmean(intEvapArray,axis=1) 
    # Sum every timestep over all windows for evap flag
    nFlagEvap = np.nansum(flagEvapArray,axis=1)

    # Evaluate the flags for dependint of the fDetect choice
    if fDetect == 'Any': # Needs one flag per timestep to consider precip or evap
        fPcp = (nFlagPcp>0) 
        fEvap = (nFlagEvap>0)
    elif fDetect == 'All': # Needs all windows flags per timestep to consider precip or evap
        fPcp = (nFlagPcp==nWindowsPerDay)
        fEvap = (nFlagEvap==nWindowsPerDay)
    elif fDetect == 'Half':# Needs Half windows flags per timestep to consider precip or evap
        fPcp = (nFlagPcp>=nWindowsPerDay/2)
        fEvap = (nFlagEvap>=nWindowsPerDay/2)

    # Fetch the precip flagged with fDetect
    intPcpFilt[~fPcp] = 0
    
    # Fetch the evap flagged with fDetect
    intEvap[~fEvap] = 0
    
    # reacumulated with filtered precipitations
    cumPcpFilt = np.nancumsum(intPcpFilt, axis=0)
    
    # Naf everything to get rid of any negative artifacts
    cumPcpFilt=NAF(cumPcpFilt,intPcpTh)

    # reacumulated with filtered evaporation
    cumEvap = np.nancumsum(intEvap,axis=0)
    # reverse Naf everything to get ride of any negative artifacts
    cumEvap=-NAF(-cumEvap,intPcpTh)


    if output_type == 'dictionary':
        out_NAF_SEG = {'time':t, 'cumPcpFilt' : cumPcpFilt, 'cumEvap' : cumEvap}
        
    elif output_type == 'dataframe':
        out_NAF_SEG = pd.DataFrame({'time':t, 'cumPcpFilt' : cumPcpFilt, 'cumEvap' : cumEvap}, index=t)
    ######################################################
    # Section not yet completed by original Matlab fonction
    # sArrays = {} 
    # sArrays['cumPcpRaw'] = cumPcpRaw
    # sArrays['cumPcpArray'] = cumPcpArray
    # sArrays['cumEvapArray'] = cumEvapArray
    # sArrays['flagPcpArray'] = flagPcpArray
    # sArrays['flagEvapArray'] = flagEvapArray #ok<*STRNU
    ####################################################
    return out_NAF_SEG

def NAF(pRaw,dpTh):
    '''
    function NAF
    
    Written by Alan Barr, 28 Aug 2012, Environment and Climate Change Canada,
    Climate Research Division, Saskatoon, Canada
    Transpose to Python by Jean-Benoit Madore Apr 2022
    
    Corresponding author: Craig D. Smith, Environment and Climate Change 
    Canada, Climate Research Division, Saskatoon, Canada
    craig.smith2@canada.ca
    
    The NAF algorithm cleans up a cumulative precipitation time series (Pcp)
    by transferring changes below a specified threshold dpTh
    to neighbouring periods,and eliminating large negative changes
    associated with gauge servicing (bucket emptying).
    
    Syntax: pNAF=NAF(pRaw,dpTh)
    
    ************************************************************************
    Inputs: pRaw, dpTh
    
    pRaw: numpy array: Measured cumulative precipitation time series derived 
    from the differential bucket weight and can have a time resolution from 
    1-minute to hourly
    
    dpTh: float: Minimum interval precipitation threshold for the filter.
          Typically set to a value between 0.05 and 0.1, depending on 
          instrument precision and uncertainty.
    ************************************************************************
    Outputs: pNAF
    
    PcpClean: numpy array: Filtered precipitation time series with the same
         temporal resolution as the input time series, pRaw
    ************************************************************************
    
    The filtering is done using a "brute-force" algorithm (Pan et al., 2015)
    that identifies small or negative changes (below dpTh)then transfers
    them to neighbouring positive changes thus aggregating all changes to
    values above dpTh. The transfers are made in ascending order,
    starting with the lowest (most negative). The cumulative total remains
    unchanged. See Smith et al. (2019) for process description
    
    Revisions:
    20181204   Amber Ross (ECCC, CRD, Saskatoon),condition added to deal with
               cases where the net accumulation is negative  
    
    References:
    
    Pan, X., Yang, D., Li, Y., Barr, A., Helgason, W., Hayashi, M., Marsh, P.,
    Pomeroy, J., and Janowicz, R. J.: Bias corrections of precipitation
    measurements across experimental sites in different ecoclimatic regions of
    western Canada, The Cryosphere, 10, 2347-2360, 
    https://doi.org/10.5194/tc-10-2347-2016, 2016.
    
    Smith, C. D., Yang, D., Ross, A., and Barr, A.: The Environment and 
    Climate Change Canada solid precipitation intercomparison data from 
    Bratt’s Lake and Caribou Creek, Saskatchewan, Earth Syst. Sci. 
    Data Discuss., https://doi.org/10.5194/essd-2018-110, in review, 2018.
    '''


    # Base the analysis on non-missing values only
    # by abstracting a sub-time series <xPcp>.

    # Find not nan values
    iYaN=np.where(~np.isnan(pRaw))[0]

    # Lenght of the serie
    nYaN=len(iYaN)
    
    # Select non-nan values within original array
    xPcp=pRaw[iYaN] 

    # Base the analysis on interval precip <dxPcp>.
    # Deaccumulate and add 0 at the begining of the array
    dxPcp = np.insert(np.diff(xPcp),0,0)

    # Eliminate servicing drops.
    # All value below -10 are considered either service or bad values
    dpServicingTh=-10
    # Find all servicing values
    itServicing=np.where(dxPcp<dpServicingTh)[0]
    
    # Delete services from precip array
    dxPcp = np.delete(dxPcp, itServicing)
    # Delete services index in the not nan index array
    iYaN =np.delete(iYaN, itServicing)

    
    #Dec 4, 2018
    #condition added to deal with cases where the net accumulation is
    #negative

    if sum(dxPcp)>dpTh:# check if data makes sens? 

        #Identify small <Drops> to be aggregated to <dpTh> or higher

        # Find all values that are both bellow the precip threshold
        # and not equal to 0 
        iDrops = np.where( (dxPcp<dpTh) & (dxPcp!=0) )[0]
        # Count the number of negative values
        nDrops=len(iDrops)
        
        #Transfer the small <Drops> one at a time,
        #and reassign small <Drops> after each transfer.
        #<Drops> are transferred to <Gets>.

        iPass=0
        while nDrops>0: 
            # Count the number of iteration during the while loop
            iPass+=1
            
            ### original Matlab print every 1000 pass. 
            ### Removed from python version
            # if mod(iPass,1000)==0;
            #     disp(sprintf('%5.0f %7.0f   %5.2f %5.2f',iPass,nDrops,nanmin(dxPcp),nanmax(dxPcp)));
            # end;

            # Lowest <dxPcp> to be eliminated. Artifact. not really used
            dxPcpN = np.min(dxPcp[iDrops])
            
            # Find index of the lowest dxPcp to be eliminated
            # Note the subset of dxPCP[iDrops]
            jDrop = np.argmin(dxPcp[iDrops])
            # Get the index within the index array iDrops that is all 
            # identified negative values
            iDrop=iDrops[jDrop]
            
            ############ METHOD FOR DROP TRANSFER ########################
            # Find nearest neighbour <Gets> to transfer <Drop> to.
            # Include in neighbour id criteria not only the distance
            # between points <d2Neighbour> but also the neighbours' <dxPcp> 
            # value (with a small weighting) to select higher <dxPcp> if two
            # neighbours are equidistant
            ###############################################################
            
            # Get all positive values
            iGets=np.where(dxPcp>0)[0]
            
            # Generate an array of all positive values excluding iDrop
            # (iDrop could be positive if iDrop< dpTh and iDrop > 0)
            iGets=np.setdiff1d(iGets,iDrop)
            
            # Count number of positive values
            nGets=len(iGets)
            
            # Absolute values of distance index around iDrop
            d2Neighbour=abs(iGets-iDrop) # number of periods apart.
            
            # Get all values identify as iGets. See above
            dxPcpNeighbour=dxPcp[iGets]
            ### [dN,jGet]=min(d2Neighbour-dxPcpNeighbour/1e5); iGet=iGets(jGet);
            
            # Lowest neighbour value. Artifact. not really used
            dN = np.min(d2Neighbour-dxPcpNeighbour/1e5)
            # Find neighbour index with precipition ponderation
            # Find the closest with lesser precip. 
            jGet = np.argmin(d2Neighbour-dxPcpNeighbour/1e5)
            # Get index in the iGets array of positive index
            iGet=iGets[jGet]            
            
            # transfer <Drop> to <Get> and set <Drop> to zero.
            
            dxPcp[iGet] = dxPcp[iGet]+dxPcp[iDrop]
            
            dxPcp[iDrop] = 0

            # reset <iDrops> and <nDrops>

            iDrops = np.where((dxPcp<dpTh) & (dxPcp!=0))[0]
            nDrops=len(iDrops)
            
#             if fEcho
#                 disp(sprintf('%5.0f %5.0f %5.0f  %7.4f %7.4f', ...
#                     [i iDrop iGet dxPcp(iDrop) dxPcp(iGet)]));
#                 if mod(jDrop,100)==0; pause; end;
#                 disp(' ');
#             end;

#         end; # while nDrops>0;

#     end; # if sum(dxPcp)>dpTh

   # print('iPass: '+ str(iPass))
    # Assign output PcpClean from sum of cleaned dxPcp values.

    # Generate empty numpy array to get the new filtered values
    pNAF = np.empty(len(pRaw))
    # put np.nan to all values
    pNAF[:] = np.nan
    # Reacumulate precipitation to not nan iYan indexes
    pNAF[iYaN] = np.nancumsum(dxPcp)
    return pNAF
    
 

######################################################################
def gapsize(x):
    '''
    [gs]=gapsize(x)
    gapsize determines the number of contiguous missing data 
    for each element of the column vector x.
    Transpose to Python by Jean-Benoit Madore Apr 2022
    '''
    # Make a nan mask based on bolean 1 or 0
    fNaN=np.isnan(x).astype(int)

    # Array of 0 of len(x)
    gs=np.zeros(np.shape(x)) 
    
    # Lenght of initial array
    lx=len(x) 

    ##### Eval gaps ########
    ### All no data as a value of 1. 
    
    # Diff of 1 means the end of gap
    # Diff of 0 means no change
    
    # Diff of 1 means the begening of a gap
    # Find all gap start
    # np.diff remove one value. +1 to adust gap start
    iGapStart = np.where(np.diff(fNaN)== 1)[0] + 1

    # Manage case where array x start with a gap
    if fNaN[0]==1: 
        iGapStart=np.append(0, iGapStart)

    # Diff of -1 means the end of gap
    # Find all gap end
    # np.diff remove one value. +1 to adust gap start
    iGapEnd = np.where(np.diff(fNaN) == -1)[0] +1
    
    # Manage case where array x end with a gap
    # Add index of last value
    if fNaN[-1]==1: 
        iGapEnd = np.append(iGapEnd, lx-1)
 
    # number of gaps
    nGaps=len(iGapStart)
    
    
    for i in range( 0, nGaps):
        # iterate through the indexes
        
        if iGapEnd[i] == iGapStart[i]: # Case the gap size is one element
            gs[iGapStart[i]] = 1
        else: # Put the number of item in the gap as value in the gs array
            gs[iGapStart[i]:iGapEnd[i]]=iGapEnd[i]-iGapStart[i]
           
    return gs

#############################################################################

def RestoreGaps(intPcpArray,flagPcpArray,nWindowsPerDay,cumPcpRaw,nt):
    '''
    function [intPcpArray,flagPcpArray] =     RestoreGaps(intPcpArray,flagPcpArray,nWindowsPerDay,cumPcpRaw,nt)

    RestoreGaps - Developed for PcpFiltPosTh24hMovingWindow 
    Due to the nature of PcpFiltPosTh24hMovingWindow, precipitation
    is not preserved during all gaps. This is dependent on the number
    of windows that are successful in preserving precipitation during 
    gaps. Two cases are examined in which gaps are restored ('All' flag):
      - at least one window preserves pcp during gap  
      - no windows preserve pcp during gap 
    
    Syntax: [intPcpArray,flagPcpArray] = 
            RestoreGaps(intPcpArray,flagPcpArray,nWindowsPerDay,cumPcpRaw,nt)
    Inputs: 
    All inputs are from PcpFiltPosTh24hMovingWindow from the NAF-SEG function
            
    Outputs: 
    intPcpArray and flagPcpArray with missed gaps restored  
    Written by Amber Ross Jan 31, 2019 
    Transpose to Python by Jean-Benoit Madore Apr 2022
    
    '''
    
    #preRow evaluate row before current iteration
    # 
    preRow = np.full((1, nWindowsPerDay), -1)
    
    # look at each row and column in intPcpArray
    # Iterate through all the row until lenght of precip (nt)
    for row, curRow in enumerate(intPcpArray):
        # Set preRow on last evaluated row. If row == 0 then preRow == curRow
        if row > 0:
            preRow = intPcpArray[row-1,:]
        # Find precip in current row
        curPosNum = np.where(curRow>0)[0]
        
        # Find no precip in current row
        curZeroNum = np.where(curRow==0)[0]
 
        # Find missing value in the last row
        preMissing = np.where(np.isnan(preRow))[0]

        # case where at least one window preserves pcp during gap  
        if (len(curPosNum)>=1) & (len(curZeroNum)>=1) & (len(preMissing)==nWindowsPerDay):

            intPcpArray[row,curZeroNum] = np.nan #% so zeros aren't included in nanmean
            flagPcpArray[row,:] = 1          # % so it is flagged as precip  
        #% case where no windows preserve pcp during gap
        elif (len(preMissing)==nWindowsPerDay) & (~np.isnan(cumPcpRaw[row])):

            endGap = cumPcpRaw[row]
            #% look backwards to find the next real number
            for z in range(0,row):

                tmpVar = cumPcpRaw[row-z]
                if ~np.isnan(tmpVar):
                    startGap = tmpVar
                    break

            jump = endGap-startGap
            if jump > 0.2 :
                intPcpArray[row,:] = jump #% so the nanmean equals the jump 
                flagPcpArray[row,:] = 1   #% so it is flagged as precip
    return intPcpArray, flagPcpArray
   