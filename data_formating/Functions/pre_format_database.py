# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:26:15 2022

@author: alexi
"""
import pandas as pd
import numpy as np
import os, glob, sys
from pathlib import Path
import re
import matplotlib.pyplot as plt
def preformat_func(source_func, data_path, save_path, hydromet_path):
    source_func(data_path, save_path, hydromet_path)
def temp_correction(subdf:pd.DataFrame,var:str,gmon_alt:float,meteo_alt:float):
    """
    Correction by the standard atmospheric laps rate (-6.5deg/km) for the temperature when the station is not the smae as the GMON one.
    Parameters
    ----------
    subdf: THe station subdf
    var: the variable look at
    gmon_alt: alt of the gmon stat
    stat_alt: altitude of the meteo station

    Returns
    -------

    """
    deltaz = meteo_alt-gmon_alt
    subdf[var] = subdf[var] - 6 * deltaz/1000

    return subdf
#%% HQ
def HQ(path, savepath, hydromet_path):
    """
    Parameters
    ----------
    path : string
        répertoire des données brutes
    
    savepath : string
        répertoire des données agrégées

    Returns
    -------
    La fonction sert à agréger tous les fichiers de données en une seule base
    de données. Dans le cas de données venant d'HydroQuébec, plusieurs bases
    de données sont crées en fonction de la présence de mesures de 
    préciptiation par pluviomètre (HQ_df_noMissingData.csv) ou non
    (HQ_df_precipMissing.csv).

    """    
    all_files = glob.glob(os.path.join(path, '*.xlsx'))
    file_names = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
    
    hydromet_files = glob.glob(os.path.join(hydromet_path, '*.xlsx'))
    
    # Correspondance entre les stations GMON et météo
    # TODO compléter le dict lorsque toutes les stations seront acquises
    st_dict = {'ARGENT': 'ROQUEMON',
               'AUXLOUPS': 'PLETIPI',
               'BAUBERT': 'BAUBERT',  # hr déjà dispo
               'BETSIA_M': 'BETSIA_M',  # hr déjà dispo
               'CABITUQG': 'LCABITUQ',  # gapfill manquant
               'CONRAD': 'MANOUA_M',  # gapfill manquant
               'DIAMAND': 'lCUTAWAY',  # gapfill manquant
               'GAREMANG': 'GAREMAND',
               'HARTJ_G': 'HART_JAU',
               'LAVAL': 'NA',
               'LACROI_G': 'LACCROIX',
               'LAFLAM_G': 'LAFLAMME',
               'LBARDO_G': 'LBARDOUX',
               'LEVASSEU': 'RLEVASSE',
               'LOUISE_G': 'LLOUISE',
               'LOUIS': 'MIQUELON',
               'MOUCHA_M': 'MOUCHA_M',  # hr déja dispo
               'NOIRS': 'OUT_4_S1',
               'PARENT_G': 'PARENT_G',  # gapfill manquant
               'PARLEUR': 'PARLEUR',  # hr déjà dispo
               'PERDRIX': 'LSTEANN2',
               'PIPMUA_G': 'PIPMUACA',
               'PORTO': 'BERS_1E1',
               'ROMA_SE': 'ROMA_SE',  # gapfill manquant
               'ROUSSY_G': 'LROUSSY',
               'RTOULNUS': 'RTOULNUS',  # hr déjà dispo
               'SAUTEREL': 'SAUTEREL',  # hr déjà dispo
               'SM3CAM_G': 'SM3CAM_G',  # gapfill manquant
               'STMARG_G': 'STEMARGC',  # hr déjà dispo
               'WABISTAN': 'WAGEGUMA',
               'WEYMOU_G': 'WEYMOUNT'
               }

    gapfill_col = ['HUMIDITE_AIR_HORAIRE (301)',
                   'NEIGE_SOL_HORAIRE (801)',
                   'PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)',
                   'TEMP_AIR_MAX_HORAIRE (204)',
                   'TEMP_AIR_MIN_HORAIRE (205)',
                   'TEMP_AIR_MOY_HORAIRE (203)',
                   "DIR_VENT_MOY60MINUTES_HORAIRE_10METRES (403)",
                   "VIT_VENT_MOY60MINUTES_HORAIRE_10METRES (503)",
                   "DIR_VENT_MOY15MINUTES_15MINUTES_2,5METRES (405)",
                   "VIT_VENT_MOY15MINUTES_15MINUTES_2,5METRES (505)", ]
    
    elev_dict = {
                'ARGENT': 641,
                'AUXLOUPS': 537.9,
                'BAUBERT': 541,
                'BETSIA_M': 403, # à valider
                'CABITUQG': 491,
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
                'NOIRS': 385,
                'PARENT_G': 442,
                'PARLEUR': 485,
                'PERDRIX': 315,
                'PIPMUA_G': 566.2,
                'PORTO': 413,
                'ROMA_SE': np.nan, 
                'ROUSSY_G': 456,
                'RTOULNUS': 688,
                'SAUTEREL': 459,
                'SM3CAM_G': 522,
                'STMARG_G': 461,
                'WABISTAN': 565,
                'WEYMOU_G': 400,
                    }
    elev_dict_stat_meteo = {
        'ROQUEMON': 641,
        'PLETIPI': 537.9,
        'BAUBERT': 541,
        'BETSIA_M': 403,  # à valider
        'LCABITUQ': 491.0,
        'CONRAD': 433,
        'MANOUA_M': 537,
        'lCUTAWAY': 440.0,
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
        'SAUTEREL': 459,
        'STEMARGC': 461,
        'SM3CAM_G': 522,
        'WAGEGUMA': 565,
        'WEYMOUNT': 363,
        'NA': np.nan, }

    print('===================')
    print('Base de données HQ')
    print('===================')
    print('Source des données:')
    print(path)
    print()
    print('Données traitées sauvegardées dans:')
    print(savepath)
    print()
    
    # Loading of data
    dataFrame = []
    
    count = 0

    for file, filename in zip(all_files, file_names):
        count += 1
        print('Station ', count)
        print(re.split("Historique_|_2", filename)[1] + ' chargée')
        
        station = re.split("Historique_|_2", filename)[1]
        df = pd.read_excel(file, parse_dates=['Horodatage (TUC)'])
        df['FILENAME'] = station
        df['source'] = 'HQ'
        df['disdrometer_type'] = 'LUFFT_WS100'
        df['pluviometer_type'] = 'PLUVIO_OTT'
        df['elevation'] = elev_dict.get(station)
        df.set_index('Horodatage (TUC)', inplace=True)
        
        # Les observations aux stations GMON de températures sont biaisées, 
        # on leur assigne NaN. Sinon, on ajoute une colonne de NaN pour les
        # types d'observations manquantes.

        df_temp_copy_tot = pd.DataFrame().reindex_like(df)

        if 'TEMP_AIR_MOY_HORAIRE (203)' in df.columns:
            df_temp_copy_tot = df_temp_copy_tot.loc[:, df_temp_copy_tot.columns.intersection(['TEMP_AIR_MOY_HORAIRE (203)'])]
            df_temp_copy_tot['TEMP_AIR_MOY_HORAIRE (203)'] = df['TEMP_AIR_MOY_HORAIRE (203)'].replace(-999.00,np.nan)

            # print(df_temp_copy_tot)

        for col in gapfill_col:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df.loc[df[col] == -999, col] = np.nan


            if col in ['TEMP_AIR_MAX_HORAIRE (204)',
                        'TEMP_AIR_MIN_HORAIRE (205)',
                        'TEMP_AIR_MOY_HORAIRE (203)']:
                df[col] = np.nan

        # Gapfill initial des données
        station_hydromet = st_dict.get(station)    
        hydromet_to_load = [file for file in hydromet_files 
                            if station_hydromet in file]

        if hydromet_to_load:
            print('Gapfill...')
            for hyd in hydromet_to_load:
                hydromet_df = pd.read_excel(hyd,
                                        parse_dates=['Horodatage (TUC)'])
                hydromet_df.set_index('Horodatage (TUC)', inplace=True)

                # num_nan_hydromet = np.sum(np.isnan(hydromet_df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)']))
                # num_nan_gmon = np.sum(np.isnan(df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)']))
                # print(num_nan_gmon,num_nan_hydromet)
                # if ('PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)' in df.columns) and ('PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)' in hydromet_df.columns) and (np.sum(hydromet_df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)'].diff()) !=0) :
                # if ('PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)' in df.columns) and (
                #         'PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)' in hydromet_df.columns) and (num_nan_hydromet <= num_nan_gmon) \
                #         and (np.sum(hydromet_df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)'].diff()) !=0) :
                #     df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)'] = np.nan


                df.fillna(hydromet_df, inplace=True)
                # print(np.sum(df['PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)'].diff()))


        # remplissage avec temp moy des gmon si manque au stations
        # correction diff altitude entre meteo et gmon
        if 'TEMP_AIR_MOY_HORAIRE (203)' in df_temp_copy_tot.columns:
            stat_meteo = st_dict[station]
            alt_gmon_stat = elev_dict[station]
            alt_meteo_stat = elev_dict_stat_meteo[stat_meteo]
            df_temp_copy_tot = temp_correction(df_temp_copy_tot, 'TEMP_AIR_MOY_HORAIRE (203)', alt_gmon_stat, alt_meteo_stat)
            df.fillna(df_temp_copy_tot, inplace=True)

        df.reset_index(inplace=True)
        dataFrame.append(df)

        print()
        
    
    # List of variables of interest
    var_names = ["Horodatage (TUC)", 
                 "FILENAME",
                 "source", "disdrometer_type", "pluviometer_type", 'elevation',
                 "LUFFT_PRECIP_TOT_MM (11786)",
                 "LUFFT_PART_TYPE (11781)",
                 "EEN_K (11335)",
                 "NEIGE_SOL_HORAIRE (801)",
                 "TEMP_AIR_MAX_HORAIRE (204)",
                 "TEMP_AIR_MIN_HORAIRE (205)",
                 "TEMP_AIR_MOY_HORAIRE (203)",
                 "HUMIDITE_AIR_HORAIRE (301)",
                 "PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)",
                 "DIR_VENT_MOY15MINUTES_15MINUTES_2,5METRES (405)",
                 "VIT_VENT_MOY15MINUTES_15MINUTES_2,5METRES (505)",
                 "DIR_VENT_MOY60MINUTES_HORAIRE_10METRES (403)",
                 "VIT_VENT_MOY60MINUTES_HORAIRE_10METRES (503)",
                 ]

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
    dataFrame = pd.concat(dataFrame, axis=0, ignore_index=True)
    
    # Join des informations par station
    dataFrame = dataFrame.sort_values(by=['Horodatage (TUC)', 'FILENAME'],
                                      kind='mergesort').sort_index()
    
    # Retrait des colonnes superflues
    for col in dataFrame.columns:
        if col not in var_names:
            dataFrame = dataFrame.drop(columns=col)
    dataFrame.rename(columns=var_dict, inplace=True)    
    
    dataFrame.reset_index(inplace=True)
    dataFrame = dataFrame.drop_duplicates(subset=['date', 'filename'])
    
    dataFrame.set_index('date')
    # Sorting des dates par stations
    for station, subdf in dataFrame.groupby('filename'):
        subdf = subdf.sort_index()
        dataFrame.loc[dataFrame['filename'] == station] = subdf
    
    dataFrame.to_csv(os.path.join(savepath, 'dataset_15min.csv'),
                     index=False)    
    
    # missingNoData = []
    # missingNoData_list = []    
        
    # missingExtraData = []
    # missingExtraData_list = []
    
    # for filename, df in zip(all_files, dataFrame):      
    #     df_filter = []
    #     print('STATION ' + re.split("Historique_|_2", filename)[1])
    #     try:
    #         df_filter = df[var_names]
    #         missingNoData.append(df_filter)
    #         missingNoData_list.append(Path(filename).stem)    
            
    #         print("Aucun type de données manquantes")
    #         print()
    #     except:
    #         missingExtraData.append(df)
    #         missingExtraData_list.append(Path(filename).stem)
    #         print("Erreur:", sys.exc_info()[1])                                
    #         print("Manque disdromètres et plus")
    #         print()
                
    # df_missingNoData = pd.concat(missingNoData, axis=0)        
    # df_missingNoData.rename(columns={
    #     "Horodatage (TUC)": "date", "FILENAME": "filename",
    #     "LUFFT_PRECIP_TOT_MM (11786)": "precip_tot_disdro",
    #     "LUFFT_PART_TYPE (11781)": "type_precip",
    #     "EEN_K (11335)": "EEN_K",
    #     "NEIGE_SOL_HORAIRE (801)": "neige_sol",
    #     "PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)": "precip_tot_pluvio",
    #     "TEMP_AIR_MAX_HORAIRE (204)": "temp_max",
    #     "TEMP_AIR_MIN_HORAIRE (205)": "temp_min",
    #     "TEMP_AIR_MOY_HORAIRE (203)": "temp_moy",
    #     "DIR_VENT_MOY15MINUTES_15MINUTES_2,5METRES (405)": "dir_vent_moy",
    #     "HUMIDITE_AIR_HORAIRE (301)": "humidite_air",         
    #     "VIT_VENT_MOY15MINUTES_15MINUTES_2,5METRES (505)": "vitesse_vent_moy"
    #     }, inplace=True)
    # df_missingNoData.to_csv(os.path.join(savepath, 'HQ_df_noMissingData.csv'),
    #                         index=False)    
    
    # var_list = ["Horodatage (TUC)", "FILENAME", "source", "disdrometer_type",
    #             "pluviometer_type",
    #             "LUFFT_PRECIP_TOT_MM (11786)", 
    #             "LUFFT_PART_TYPE (11781)",
    #             "EEN_K (11335)",
    #             "HUMIDITE_AIR_HORAIRE (301)",
    #             "NEIGE_SOL_HORAIRE (801)",
    #             "TEMP_AIR_MOY_HORAIRE (203)",
    #             "TEMP_AIR_MAX_HORAIRE (204)",
    #             "TEMP_AIR_MIN_HORAIRE (205)"
    #             ]
    # df_missingData = pd.concat(missingExtraData, axis=0)
    # df_missingData = df_missingData[var_list]    
    
    # df_missingData.rename(columns={
    #     "Horodatage (TUC)": "date", "FILENAME": "station",
    #     "LUFFT_PRECIP_TOT_MM (11786)": "precip_tot_disdro",
    #     "LUFFT_PART_TYPE (11781)": "type_precip",
    #     "EEN_K (11335)": "EEN_K",        
    #     "NEIGE_SOL_HORAIRE (801)": "neige_sol",
    #     "PRECIPITATION_TOTALE_OBSERVE_15MINUTES (604)": "precip_tot_pluvio",
    #     "TEMP_AIR_MOY_HORAIRE (203)": 'temp_moy',
    #     "TEMP_AIR_MAX_HORAIRE (204)": "temp_max",
    #     "TEMP_AIR_MIN_HORAIRE (205)": "temp_min",
    #     "HUMIDITE_AIR_HORAIRE (301)": "humidite_air",         
    #     }, inplace=True)
        
    # df_missingData[['precip_tot_pluvio', 'dir_vent_moy',
    #                   'humidite_air', 'vitesse_vent_moy']] = np.nan
    # df_missingData['pluviometer_type'] = 'N/A'
    # df_missingData.to_csv(os.path.join(savepath, 'HQ_df_missingData.csv'),
    #                      index=False)
    
    # # Création d'une liste compilant la disponibilité des données des stations
    # print('Stations avec mesures de précipitomètre: \n')
    # print(list(df_missingNoData['station']
    #            .unique()))
    
    # print('Stations manquant plusieurs types de données: \n')
    # print(list(df_missingData['station']
    #            .unique()))
    
    print('Pré-formatage des données HQ terminé')
    print('====================================')
     
    return

#%%
# TODO pré formatage UQAM

#%%
# TODO pré formatage SN et SAP
