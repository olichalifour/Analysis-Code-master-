# -*- coding: utf-8 -*-
"""
@author: Olivier
"""

import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import numpy as np
from Functions.formating_functions import time_filter
import warnings
from scipy import stats
from Function_oli.foret_momo import standard_prcp_type,fz_correction_temp
from Functions.formating_functions import NAF_SEG
from Functions.undercatch import catch_effic_eq4
import matplotlib.pyplot as plt

def creation_dataset_uqam(path_gen: str, saving_path: str):
    """
    Creat the CSV for the site neige station

    Parameters
    ----------
    path_gen : general path containing all the data needed
    saving_path: path of where we want t save the data

    Returns
    -------

    """

    list_dataframe_uqam = []
    fct_uqam = [meta_uqam, parsivel_uqam]
    # fct_uqam = [parsivel_uqam]
    for fct in fct_uqam:
        df = fct(path_gen)
        list_dataframe_uqam.append(df)

    df_total_uqam = pd.concat(list_dataframe_uqam, axis=1)
    df_total_uqam.sort_index(inplace=True)

    # correction freezing rain if PR & TT<0 -> FZ
    df_total_uqam = fz_correction_temp(df_total_uqam)

    # threshold geonor obs
    df_total_uqam.loc[df_total_uqam['precip_inst_pluvio'] < 0.1, ['precip_inst_pluvio']] = 0

    # threshold geonor
    df_total_uqam.loc[df_total_uqam['precip_inst_pluvio'] > 75, ['precip_inst_pluvio']] = np.nan

    df_total_uqam.to_csv(os.path.join(saving_path, 'dataset_15min_total_formated.csv'))







    pdt_15min_to_1h(df_total_uqam, saving_path)


def meta_uqam(path_gen: str) -> pd.DataFrame:
    """
    Open the Metadata files for the UQAM stations and treat PR,HR,Wind,TEMP.

    Parameters
    ----------
    path_gen

    Returns
    -------
    The metadata dataframe
    """
    print('______')
    print('meta treatment PK uqam')

    all_files_1 = sorted(glob.glob(os.path.join(path_gen, 'meteo_pk', 'data_1', '*')))
    all_files_2 = sorted(glob.glob(os.path.join(path_gen, 'meteo_pk', 'data_2', '*')))
    all_files_3 = sorted(glob.glob(os.path.join(path_gen, 'meteo_pk', 'server_file', '*')))
    all_files_4 = sorted(glob.glob(os.path.join(path_gen, 'meteo_pk', 'data_2_prcp', '*')))
    list_df_meta = []

    # open every monthly file and creat the 1 min dataframe.


    for file_1 in all_files_1:
        dict_col = {"TIMESTAMP": "date",
                    'Temp_1m': 'temp_moy',
                    'Humidite_1m': 'humidite_air',
                    'GEONOR_PCPN_Moyenne_3': "precip_inst_pluvio",
                    'Vent_vitesse_2m': 'vitesse_vent_moy_2_5m',
                    'Vent_Dir_2m': 'dir_vent_moy_2_5m',
                    'Vent_vitesse_10m': 'vitesse_vent_moy_10m',
                    'Vent_Dir_10m': 'dir_vent_moy_10m'}

        df_meta_1 = pd.read_csv(file_1, delimiter=',', header=0)

        for col in df_meta_1.columns:
            if col not in dict_col.keys():
                df_meta_1.drop(columns=col, axis=1, inplace=True)

        df_meta_1.rename(columns=dict_col, inplace=True)
        df_meta_1.set_index('date', drop=True, inplace=True)
        df_meta_1.index = pd.to_datetime(df_meta_1.index, infer_datetime_format=True)
        df_meta_1.sort_index(inplace=True)

        df_meta_1 = df_meta_1.astype({'humidite_air': 'float',
                                      'temp_moy': 'float',
                                      'dir_vent_moy_2_5m': 'float',
                                      'vitesse_vent_moy_2_5m': 'float',
                                      'dir_vent_moy_10m': 'float',
                                      'vitesse_vent_moy_10m': 'float', })

        resample_15m_dict = {
            'humidite_air': 'mean',
            'temp_moy': 'mean',
            'dir_vent_moy_2_5m': 'mean',
            'vitesse_vent_moy_2_5m': 'mean',
            'dir_vent_moy_10m': 'mean',
            'vitesse_vent_moy_10m': 'mean',
            'precip_inst_pluvio': 'sum',
        }

        # temp_uqam_format(df_meta_1)
        list_df_meta.append(df_meta_1)

    for file_2 in all_files_2:
        dict_col = {"DateTime(UTC)": "date",
                    'Temperature': 'temp_moy',
                    'Mean_precip': "precip_inst_pluvio",
                    'Humidite': 'humidite_air',
                    'VitesseVent': 'vitesse_vent_moy_10m',
                    'DirectionVent': 'dir_vent_moy_10m', }

        df_meta_2 = pd.read_csv(file_2, delimiter=',', header=0)

        for col in df_meta_2.columns:
            if col not in dict_col.keys():
                df_meta_2.drop(columns=col, axis=1, inplace=True)

        df_meta_2.rename(columns=dict_col, inplace=True)
        df_meta_2.set_index('date', drop=True, inplace=True)
        df_meta_2.index = pd.to_datetime(df_meta_2.index, infer_datetime_format=True)
        df_meta_2.sort_index(inplace=True)

        df_meta_2 = df_meta_2.astype({'humidite_air': 'float',
                                      "precip_inst_pluvio": 'float',
                                      'temp_moy': 'float',
                                      'dir_vent_moy_10m': 'float',
                                      'vitesse_vent_moy_10m': 'float',
                                      })

        df_meta_2['dir_vent_moy_2_5m'] = np.nan
        df_meta_2['vitesse_vent_moy_2_5m'] = np.nan

        # temp_uqam_format(df_meta_2)
        list_df_meta.append(df_meta_2)

    for file_3 in all_files_3:

        dict_col = {"Timestamp": "date",
                    'HMP_AirTemp': 'temp_moy',
                    'HMP_RH': 'humidite_air',
                    'RM_WindSpeed_avg': 'vitesse_vent_moy_10m',
                    'RM_WindDir_avg': 'dir_vent_moy_10m', }

        df_meta_3 = pd.read_csv(file_3, delimiter=',', header=0)

        for col in df_meta_3.columns:
            if col not in dict_col.keys():
                df_meta_3.drop(columns=col, axis=1, inplace=True)

        df_meta_3.rename(columns=dict_col, inplace=True)
        df_meta_3.set_index('date', drop=True, inplace=True)
        df_meta_3.index = pd.to_datetime(df_meta_3.index, infer_datetime_format=True)
        df_meta_3.sort_index(inplace=True)

        df_meta_3 = df_meta_3.astype({'humidite_air': 'float',
                                      'temp_moy': 'float',
                                      'dir_vent_moy_10m': 'float',
                                      'vitesse_vent_moy_10m': 'float', })

        df_meta_3['dir_vent_moy_2_5m'] = np.nan
        df_meta_3['vitesse_vent_moy_2_5m'] = np.nan

        list_df_meta.append(df_meta_3)

    for file_4 in all_files_4:

        dict_col = {"DateTime(UTC)": "date",
                    'Mean_precip': 'precip_inst_pluvio'}

        df_meta_4 = pd.read_csv(file_4, delimiter=',', header=0)

        for col in df_meta_4.columns:
            if col not in dict_col.keys():
                df_meta_4.drop(columns=col, axis=1, inplace=True)

        df_meta_4.rename(columns=dict_col, inplace=True)
        df_meta_4.set_index('date', drop=True, inplace=True)
        df_meta_4.index = pd.to_datetime(df_meta_4.index, infer_datetime_format=True)
        df_meta_4.sort_index(inplace=True)

        df_meta_4 = df_meta_4.astype({'precip_inst_pluvio': 'float', })

        list_df_meta.append(df_meta_4)

    print('\nloading done')
    print('__________')
    print('data treatment\n')

    df_meta_tot = pd.concat(list_df_meta, axis=0)

    df_meta_tot.index = pd.to_datetime(df_meta_tot.index, infer_datetime_format=True)
    df_meta_tot.sort_index(inplace=True)
    df_meta_tot = df_meta_tot[~df_meta_tot.index.duplicated(keep='first')]


    df_meta_tot = time_filter(df_meta_tot)
    df_meta_tot['precip_tot_pluvio'] = df_meta_tot['precip_inst_pluvio'].cumsum()


    print('PRCP\n')
    df_meta_tot = func_traitement_prcp(df_meta_tot)

    df_meta_tot_15min = df_meta_tot.resample('15T', label='left').agg(resample_15m_dict)
    df_meta_tot_15min['precip_tot_pluvio'] = df_meta_tot_15min['precip_inst_pluvio'].cumsum()

    print('HR\n')
    df_meta_tot_15min = hr_uqam(df_meta_tot_15min)
    print('Temperature\n')
    df_meta_tot_15min = temp_uqam_format(df_meta_tot_15min)
    print('wind\n')
    df_meta_tot_15min = wind_uqam(df_meta_tot_15min)

    return df_meta_tot_15min


def temp_uqam_format(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    get min max temperture every 15 mins and add the columns temp_min and temp_max

    Parameters
    ----------
    df_meta

    Returns
    -------

    """
    mu = df_meta['temp_moy'].mean()
    sig = 3 * df_meta['temp_moy'].std()

    df_meta.loc[(df_meta['temp_moy'] > mu + sig) |
                (df_meta['temp_moy'] < mu - sig),
                'temp_moy'] = np.nan
    df_meta.loc[(df_meta['temp_moy'] > 40) |
                (df_meta['temp_moy'] < -50),
                'temp_moy'] = np.nan

    diff = df_meta['temp_moy'].diff()
    mu = diff.mean()
    sig = 3 * diff.std()
    df_meta.loc[(diff > mu + sig) |
                (diff < mu - sig),
                'temp_moy'] = np.nan

    df_meta['temp_moy'].interpolate(method='linear', limit=9, inplace=True)

    df_meta['temp_max'] = df_meta['temp_moy'].resample('15T', label='left').agg(np.max)
    df_meta['temp_min'] = df_meta['temp_moy'].resample('15T', label='left').agg(np.min)

    return df_meta




def func_traitement_prcp(df_prcp):
    """
        Correct the precip data PR < 0 and PR > PR_mean + 3* PR_std_dev

        Applied the filter over the accumulation data and correct the udercath of the single alter shield
        Parameters
        ----------
        df_meta

        Returns
        -------

        """
    list_var_acc_tot = ['precip_tot_pluvio']
    dict_precip = {'precip_tot_pluvio': 'precip_inst_pluvio'}


    for cummu in list_var_acc_tot:
        inst = dict_precip[cummu]
        df_prcp.loc[df_prcp[cummu] < 0, cummu] = np.nan
        df_prcp[cummu] += abs(df_prcp[cummu].min())

        diff = (df_prcp[cummu]
                .fillna(method='bfill',limit=12)
                .diff())
        diff_mask = (diff < 0) | (diff >= (diff.mean() + 3 * diff.std()))
        df_prcp.loc[diff_mask, cummu] = np.nan


        df_prcp[cummu] = df_prcp[cummu].fillna(method='bfill')


        # Raccrochage des données

        # diff = (df_prcp[cummu]
        #         .fillna(method='bfill', limit=12)
        #         .diff())
        # plt.plot(df_prcp.index, df_prcp[inst].cumsum())
        df_prcp[inst] = df_prcp[cummu].diff()
        df_prcp.loc[(df_prcp[inst] < -2) | (df_prcp[inst] > 10), inst] = 0
        df_prcp[cummu] = df_prcp[inst].cumsum()

        diff = (df_prcp[cummu]
                .fillna(method='bfill', limit=12)
                .diff())
        # plt.plot(df_prcp.index, df_prcp[inst].cumsum())
        # plt.show()
        # diff_mask = (diff < 0)
        #
        # pdt = pd.to_timedelta("00:01:00")
        #
        # window_end = df_prcp.loc[diff_mask, cummu].index
        # window_start = window_end - 5 * pdt  # testé pour 10*pdt, quantité à généraliser
        #
        #
        # for i in range(len(window_end)):
        #     # applique la correction seulement si window_start fait parti de l'index
        #
        #     if window_start[i] > df_prcp.index.min():
        #
        #         corr = (-diff
        #                 .loc[window_start[i]:window_end[i]]
        #                 .min())
        #
        #         df_prcp.loc[window_end[i]:, cummu] += corr


        # Retrait des "jumps"
        # jumps = diff.copy()
        # plt.plot(df_prcp[cummu])
        # jumps.loc[(jumps < jumps.mean() + 3 * jumps.std()) |
        #           (jumps < 3.33)] = 0

        # jumps = np.cumsum(jumps)

        # df_prcp[cummu] = df_prcp[cummu] - jumps


        year_list = (df_prcp
                     .index
                     .year
                     .unique()
                     .sort_values())

        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')

            df_prcp.loc[((df_prcp.index >= start) & (
                        df_prcp.index <= end)), cummu] = df_prcp.loc[start: end,cummu] - df_prcp.loc[start: end,cummu].min()




        df_prcp[inst] = df_prcp[cummu].diff()




        # Retrait de précip instantanées négatives ou aberrantes
        # df_prcp.loc[((df_prcp[inst] < 0) | (df_prcp[inst] > df_prcp[inst].mean() + 3 * df_prcp[inst].std())),
        #        inst] = 0

        for year in df_prcp.index.year.unique()[:-1]:
            start = (str(year) + '-10-01')
            idx = df_prcp.loc[df_prcp.index >= start,
                            inst].first_valid_index()

            df_prcp.loc[(df_prcp.index == idx) ,inst] = 0

        if sum(df_prcp[cummu].isna()) < len(df_prcp[inst]):

            xt = df_prcp.index
            xRawCumPcp = df_prcp[cummu]  # Réaccumulation
            intPcpTh = 0.001
            nRecsPerDay = 24 * 60
            nWindowsPerDay = 3
            output_type = 'dataframe'

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore',
                                      category=RuntimeWarning)
                out_NAF = NAF_SEG(xt, xRawCumPcp, intPcpTh, nRecsPerDay,
                                  nWindowsPerDay, output_type)

            df_prcp[cummu] = out_NAF['cumPcpFilt']
            df_prcp[inst] = out_NAF['cumPcpFilt'].diff()




    return df_prcp
def hr_uqam(df_meta: str) -> pd.DataFrame:
    """
    Correct the Relative Humidity HR >100 and  HR <10 are remove
    Parameters
    ----------
    df_meta

    Returns
    -------

    """
    df_meta.loc[(df_meta['humidite_air'] > 100), 'humidite_air'] = np.nan
    df_meta.loc[(df_meta['humidite_air'] < 10), 'humidite_air'] = np.nan

    return df_meta


def wind_uqam(df_meta: str) -> pd.DataFrame:
    """
    Corrected the wind by removing anormal value w_speed<0 and w_direction >360 and  w_direction < 0
    Parameters
    ----------
    df_meta

    Returns
    -------

    """
    df_meta.loc[(df_meta['vitesse_vent_moy_2_5m'] < 0)|(df_meta['vitesse_vent_moy_2_5m'] > 150/3.6), 'vitesse_vent_moy_2_5m'] = np.nan
    df_meta.loc[(df_meta['vitesse_vent_moy_10m'] < 0)|(df_meta['vitesse_vent_moy_10m'] > 150/3.6), 'vitesse_vent_moy_10m'] = np.nan
    df_meta.loc[(df_meta['dir_vent_moy_2_5m'] < 0) | (df_meta['dir_vent_moy_2_5m'] > 360), 'dir_vent_moy_2_5m'] = np.nan
    df_meta.loc[(df_meta['dir_vent_moy_10m'] < 0) | (df_meta['dir_vent_moy_10m'] > 360), 'dir_vent_moy_10m'] = np.nan

    return df_meta


def parsivel_uqam(path_gen: str) -> pd.DataFrame:
    """
    get the data from the Parsivel UQAM files.

    Parameters
    ----------
    path_gen

    Returns
    -------

    """
    print('______')
    print('parsivel treatment PK uqam')

    all_files_1_parsi = sorted(glob.glob(os.path.join(path_gen, 'parsivel', 'UQAM_Parsivel_Oct2020-May2021', '*')))

    all_files_2_parsi = sorted(glob.glob(os.path.join(path_gen, 'parsivel', 'server_file', '*')))
    all_files_3_parsi = sorted(glob.glob(os.path.join(path_gen, 'parsivel', 'UQAM_disdrometer_oct2021-apr2022', '*')))

    list_df_parsi = []

    for file_1 in all_files_1_parsi:

        dict_col = {"Timestamp": "date",
                    'Weather code SYNOP WaWa': 'type_precip', }

        data = open(file_1, encoding='iso-8859-1')
        lines = data.read().splitlines()

        list_line = []
        for line in lines[1:]:
            if len(line) > 0:
                split_line = line.split('<')
                line_add = split_line[0].split(';')

                if line[0] != 'D':
                    list_line.append(line_add[:-1])

        df_parsi_1 = pd.DataFrame(list_line,
                                  columns=lines[0].split(';')[:-1])
        list_date = []
        for date, time in zip(df_parsi_1['Date'].values, df_parsi_1['Time'].values):
            datetime = date + ' ' + time

            datetime = pd.to_datetime(datetime, format='%d.%m.%Y %H:%M:%S')
            list_date.append(datetime)

        df_parsi_1['Timestamp'] = list_date
        df_parsi_1.drop('Date', axis=1, inplace=True)
        df_parsi_1.drop('Time', axis=1, inplace=True)

        for col in df_parsi_1.columns:
            if col not in dict_col.keys():
                df_parsi_1.drop(columns=col, axis=1, inplace=True)

        df_parsi_1.rename(columns=dict_col, inplace=True)
        df_parsi_1.set_index('date', drop=True, inplace=True)
        df_parsi_1.sort_index(inplace=True)

        df_parsi_1 = df_parsi_1.astype({'type_precip': 'float'})

        list_df_parsi.append(df_parsi_1)

    for file_2 in all_files_2_parsi:

        dict_col = {"Timestamp": "date",
                    'Weather_code_SYNOP_WaWa': 'type_precip', }

        df_parsi_2 = pd.read_csv(file_2, delimiter=',', header=0)

        for col in df_parsi_2.columns:
            if col not in dict_col.keys():
                df_parsi_2.drop(columns=col, axis=1, inplace=True)

        df_parsi_2.rename(columns=dict_col, inplace=True)
        df_parsi_2.set_index('date', drop=True, inplace=True)
        df_parsi_2.index = pd.to_datetime(df_parsi_2.index, infer_datetime_format=True)
        df_parsi_2.sort_index(inplace=True)

        df_parsi_2 = df_parsi_2.astype({'type_precip': 'float'})

        list_df_parsi.append(df_parsi_2)

    for file_3 in all_files_3_parsi:

        dict_col = {"Timestamp": "date",
                    'Weather code SYNOP WaWa': 'type_precip', }

        data = open(file_3, encoding='iso-8859-1')
        lines = data.read().splitlines()

        list_line = []
        for line in lines[1:]:
            if len(line) > 0:
                split_line = line.split('<')
                line_add = split_line[0].split(';')

                if line[0] != 'D':
                    list_line.append(line_add[:-1])

        df_parsi_3 = pd.DataFrame(list_line,
                                  columns=lines[0].split(';')[:-1])
        list_date = []
        for date, time in zip(df_parsi_3['Date'].values, df_parsi_3['Time'].values):
            datetime = date + ' ' + time

            datetime = pd.to_datetime(datetime, format='%d.%m.%Y %H:%M:%S')
            list_date.append(datetime)

        df_parsi_3['Timestamp'] = list_date
        df_parsi_3.drop('Date', axis=1, inplace=True)
        df_parsi_3.drop('Time', axis=1, inplace=True)

        for col in df_parsi_3.columns:
            if col not in dict_col.keys():
                df_parsi_3.drop(columns=col, axis=1, inplace=True)

        df_parsi_3.rename(columns=dict_col, inplace=True)
        df_parsi_3.set_index('date', drop=True, inplace=True)
        df_parsi_3.sort_index(inplace=True)

        df_parsi_3 = df_parsi_3.astype({'type_precip': 'float'})

        list_df_parsi.append(df_parsi_3)

    df_parci_tot = pd.concat(list_df_parsi, axis=0)
    df_parci_tot.index = pd.to_datetime(df_parci_tot.index, infer_datetime_format=True)
    df_parci_tot.sort_index(inplace=True)

    resample_15m_dict = {
        'type_precip': mode}
    df_15min_parsivel = df_parci_tot.resample('15T', label='left').agg(resample_15m_dict)

    # standarize the prcp type and correct freezing rain
    df_15min_parsivel = standard_prcp_type(df_15min_parsivel)

    return df_15min_parsivel


def pdt_15min_to_1h(df: pd.DataFrame, save_path: str):
    """
    Resample the dataframe 15 min -> 1h

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    print("Création de la base de données horaire...")

    resample_1h_op = {
        'humidite_air': 'mean',
        'temp_moy': 'mean',
        'temp_max': 'max',
        'temp_min': 'min',
        'precip_inst_pluvio': 'sum',
        "dir_vent_moy_2_5m": 'mean',
        "dir_vent_moy_10m": 'mean',
        "vitesse_vent_moy_10m": 'mean',
        'vitesse_vent_moy_2_5m': 'mean'}

    print('Resampling...')

    resamp, frac_list, frac_list_qty = phase_frac(df)

    # mask = resamp['tot'] > 0
    df_1h = df.resample('1H', label='left').agg(resample_1h_op)
    df_1h.loc[df_1h['precip_inst_pluvio'] < 0.2, ['precip_inst_pluvio']] = 0
    df_1h.loc[df_1h['precip_inst_pluvio'] > 110, ['precip_inst_pluvio']] = np.nan
    # mask = subdf_1h_mask['precip_inst_pluvio'] > 0

    # for frac in frac_list:
    #     # print(frac,np.sum(resamp.loc[mask, frac]))
    #     resamp.loc[mask, frac] = resamp.loc[mask, frac] / subdf_1h_mask.loc[mask, 'precip_inst_pluvio']
    #
    # resamp.drop(columns=['tot'], inplace=True)
    #
    # df_30min_stat = df.resample('0.5H', label='right').agg(resample_1h_op)
    #
    # df_30min_stat.loc[df_30min_stat['precip_inst_pluvio'] < 0.14, ['precip_inst_pluvio']] = 0

    # subdf_1h_stat = df_30min_stat.resample('1H', label='left').agg(resample_1h_op).loc[:mask.index[-1]]

    # for frac in frac_list:
    #     resamp.loc[mask, frac] = resamp.loc[mask, frac] * subdf_1h_stat.loc[mask, 'precip_inst_pluvio']
    #     print(frac, np.sum(resamp.loc[mask, frac]))

    df_1h = pd.concat([df_1h, resamp[frac_list + frac_list_qty]], axis=1)
    df_1h['precip_tot_pluvio'] = df_1h['precip_inst_pluvio'].cumsum()

    # undercatchment
    lidx = df_1h['precip_tot_pluvio'].last_valid_index()
    unprcp = df_1h.loc[lidx, 'precip_tot_pluvio']

    mask = df_1h['precip_inst_pluvio'] > 0

    for frac in frac_list:

        df_1h.loc[mask, frac] = df_1h.loc[mask, frac] / df_1h.loc[mask, 'precip_inst_pluvio']

    df_1h = sous_captation_df_10m(df_1h)

    for frac in frac_list:

        df_1h.loc[mask, frac] = df_1h.loc[mask, frac] * df_1h.loc[mask, 'precip_inst_pluvio']

    toprcp = df_1h.loc[lidx, 'precip_tot_pluvio']

    print(f'UQAM Diff touch-untouch: {toprcp - unprcp:.2f} mm')
    print(f'UQAM Diff touch-untouch: {((toprcp - unprcp) / unprcp) * 100:.2f} %')


    print('Sauvegarde de la base de donnée horaire...')
    df_1h.to_csv(os.path.join(save_path, 'dataset_1h.csv'))


def phase_frac(df_phase) -> (pd.DataFrame, list, list):
    """
    resample the phase fraction for each hours
    Parameters
    ----------
    df_phase

    Returns
    -------

    """

    frac_list = []
    frac_list_qty = []
    phase_list = (df_phase['type_precip'].unique())

    for phase in phase_list:

        if np.isnan(phase):
            df_phase.loc[df_phase['type_precip'].isna(),
                         f'#{phase}_1h'] = 1
            frac_list_qty.append(f'#{phase}_1h')

            phase_col = 'f_' + str(phase)
            df_phase.loc[df_phase['type_precip'].isna(),
                         phase_col] =df_phase.loc[df_phase['type_precip'].isna(),
                                                   'precip_inst_pluvio']
            frac_list.append(phase_col)

        else:
            phase_col = 'f_' + str(int(phase))
            df_phase[phase_col] = np.nan
            frac_list.append(phase_col)

            df_phase.loc[df_phase['type_precip'] == phase,
                         phase_col] = df_phase.loc[df_phase['type_precip'] == phase,
                                                   'precip_inst_pluvio']

    resamp_phase = df_phase[frac_list + frac_list_qty].resample('1H', label='left').sum()

    resamp_phase['tot'] = resamp_phase[frac_list].sum(axis=1)

    resamp_phase.drop(columns=['tot'], inplace=True)

    resamp_phase.fillna(0, inplace=True)

    return resamp_phase, frac_list, frac_list_qty


def mode(xs) -> float:
    """
    mode used for the type of precip resampling

    if all zeros array -> 0

    if mode = zeros but at least one value is non zeros in the array -> mode of the array without the zeros value in it
    non zeros = at least 1 other type detected in the array

    if mode != 0 -> mode

    Parameters
    ----------
    xs

    Returns
    -------

    """

    count = (xs == 0).sum()

    if count == len(xs.values):
        try:
            return stats.mode(xs)[0][0]
        except IndexError:
            return np.nan

    else:
        try:
            return stats.mode(xs.loc[~(xs == 0)])[0][0]
        except IndexError:
            return np.nan
def catch_effic_eq3(wind, temp, a, b, c) -> float:
    """
    Catch efficientcy
    Kochendorfer et all. 2017 eq. (3)
    Parameters
    ----------
    wind: wind speed
    temp: temperature at the station
    a: parameter from the article Kochendorfer et all. 2017
    b: parameter from the article Kochendorfer et all. 2017
    c: parameter from the article Kochendorfer et all. 2017

    Returns
    -------

    """
    CE = np.e ** (- a * wind * (1 - np.arctan(b * (temp)) + c))

    return CE

# def sous_captation_df_10m(df_capt: pd.DataFrame) -> pd.DataFrame:
#     """
#     Kochendorfer et all. 2017 for wind at 10 m
#     Parameters
#     ----------
#     df_capt
#     Returns
#     -------
#     """
#
#     times = df_capt.index
#     for i, time in enumerate(times):
#         row = df_capt.loc[time]
#         row_temp = row['temp_moy']
#         # deja en m/s a momo
#         row_wind = row['vitesse_vent_moy_10m']
#         row_prcp = row['precip_inst_pluvio']
#
#         # min of prcp and temp not none
#         if row_prcp >= 0.1:
#             # solid
#             if row_temp < -2:
#                 # CTE from Kochendorfer et all. 2017 table 2
#                 a, b, c = 0.728, 0.230, 0.336
#                 if row_wind >= 7.2:
#                     row_wind = 7.2
#                 else:
#                     row_wind = row_wind
#                 CE_row = catch_effic_eq4(row_wind, a, b, c)
#                 # if CE_row > 1:
#                 #     print(CE_row, row_type, row_temp, row_wind, row_prcp)
#                 df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio'] / CE_row
#
#             # mix
#             elif row_temp >= -2 and row_temp <= 2:
#                 a, b, c = 0.668, 0.132, 0.339
#                 if row_wind >= 7.2:
#                     row_wind = 7.2
#                 else:
#                     row_wind = row_wind
#                 CE_row = catch_effic_eq4(row_wind, a, b, c)
#
#
#                 df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio'] / CE_row
#
#
#             else:
#                 df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']
#
#         else:
#
#             df_capt.loc[time, ['precip_inst_pluvio','f_0', 'f_60', 'f_67', 'f_69', 'f_70','f_nan']] = 0
#
#     df_capt['precip_tot_pluvio'] = df_capt['precip_inst_pluvio'].cumsum()
#
#     return df_capt

def sous_captation_df_10m(df_capt: pd.DataFrame) -> pd.DataFrame:
    """
    Kochendorfer et all. 2017 for wind at 10 m
    Parameters
    ----------
    df_capt

    Returns
    -------

    """

    times = df_capt.index
    for i, time in enumerate(times):
        row = df_capt.loc[time]
        row_temp = row['temp_moy']
        # deja en m/s a momo
        row_wind = row['vitesse_vent_moy_10m']
        row_prcp = row['precip_inst_pluvio']

        # min of prcp and temp not none
        if row_prcp >= 0.22 and not np.isnan(row_temp) and row_temp <= 5:

            # CTE from Kochendorfer et all. 2017 table 2
            a, b, c = 0.0281 ,1.628 ,0.837

            # wind threshold
            if row_wind >= 9:
                row_wind = 9
            else:
                row_wind = row_wind
            CE_row = catch_effic_eq3(row_wind, row_temp, a, b, c)

            if CE_row > 1:
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']
            else:
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio'] / CE_row
            #     print(CE_row, row_type, row_temp, row_wind, row_prcp)


        else:

            df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']

    df_capt['precip_tot_pluvio'] = df_capt['precip_inst_pluvio'].cumsum()


    return df_capt
