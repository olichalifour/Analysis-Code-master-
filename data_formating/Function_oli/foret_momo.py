# -*- coding: utf-8 -*-
"""
Created by: Olivier Chalifour
Date: 20-09-2022

"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
import matplotlib.pyplot as plt
from Functions.formating_functions import NAF_SEG
from scipy import stats


def creation_dataset_momorency(path_gen: str, saving_path: str):
    """
    Creat the CSV for the site neige station


    Parameters
    ----------
    path_gen : general path containing all the data needed
    saving_path: path of where we want t save the data

    Returns
    -------

    """

    list_dataframe_momo = []
    fct_momo = [temp_momo_forest, prcp_momo_forest, wind_momo_forest, parsivel_momo_forest]

    for fct in fct_momo:
        df_var = fct(path_gen, saving_path)
        list_dataframe_momo.append(df_var)

    df_total_momo = pd.concat(list_dataframe_momo, axis=1)

    df_total_momo.sort_index(inplace=True)

    df_total_momo = fz_correction_temp(df_total_momo)



    # threshold geonor
    df_total_momo.loc[df_total_momo['precip_inst_pluvio'] > 75, ['precip_inst_pluvio']] = np.nan
    df_total_momo.loc[df_total_momo['precip_inst_geonor'] > 75, ['precip_inst_geonor']] = np.nan

    # threshold geonor obs
    df_total_momo.loc[df_total_momo['precip_inst_pluvio'] < 0.1, ['precip_inst_pluvio']] = 0
    df_total_momo.loc[df_total_momo['precip_inst_geonor'] < 0.1, ['precip_inst_geonor']] = 0


    df_total_momo.to_csv(os.path.join(saving_path, 'dataset_15min_total_formated.csv'))

    pdt_15min_to_1h(df_total_momo, saving_path)


def temp_momo_forest(path_gen: str, saving_path: str) -> pd.DataFrame:
    """
    function that return the temperature with a dt of 15 mins. for the temperature and humidity, the average is calculated every 15 mins.
    Parameters
    ----------
    path_gen: general path where the temperature files are located
    saving_path: saving path where the results are save
    Returns:
    Dataframe with the temperature with dt = 15 mins
    -------

    """

    # check is file is already done

    print('______')
    print('Temp treatment Foret Montmorency')

    if not os.path.isfile(os.path.join(saving_path, 'variables', 'temp_dataset_15min.csv')):

        all_files = sorted(glob.glob(os.path.join(path_gen, 'HMP', '*')))

        list_df_temp = []

        # open every monthly file and creat the 1 min dataframe.
        # The extreme are remove if value is 3 time sigma from the mean value

        for file in all_files:
            df_1month = pd.read_csv(file, delimiter=';', header=0)

            dict_col = {"TIMESTAMP": "date", 'AirTC_Avg': 'temp_moy', 'RH': 'humidite_air'}
            df_1month.rename(columns=dict_col, inplace=True)

            df_1month.set_index("date", drop=True, inplace=True)
            df_1month.index = pd.to_datetime(df_1month.index, infer_datetime_format=True)
            df_1month.sort_index(inplace=True)

            df_1month.drop(columns='RECORD', axis=1, inplace=True)

            df_1month.loc[df_1month['humidite_air'] > 100, 'humidite_air'] = np.nan

            # extreme treatement

            mu = df_1month['temp_moy'].mean()
            sig = 3 * df_1month['temp_moy'].std()

            df_1month.loc[(df_1month['temp_moy'] > mu + sig) |
                          (df_1month['temp_moy'] < mu - sig),
                          'temp_moy'] = np.nan
            df_1month.loc[(df_1month['temp_moy'] > 40) |
                          (df_1month['temp_moy'] < -50),
                          'temp_moy'] = np.nan

            df_1month['temp_moy'].interpolate(method='linear', limit=9, inplace=True)

            list_df_temp.append(df_1month)

        df_temp_1min = pd.concat(list_df_temp)
        df_temp_1min.sort_index(inplace=True)

        resample_15m_dict = {
            'humidite_air': 'mean',
            'temp_moy': 'mean',
        }

        # make dataframe 15 mins
        # get min and max every 15 min

        df_15min_temp = df_temp_1min.resample('15T', label='left').agg(resample_15m_dict)
        df_15min_temp['temp_max'] = df_temp_1min['temp_moy'].resample('15T', label='left').agg(np.max)
        df_15min_temp['temp_min'] = df_temp_1min['temp_moy'].resample('15T', label='left').agg(np.min)

        df_15min_temp.to_csv(os.path.join(saving_path, 'variables', 'temp_dataset_15min.csv'))

    else:
        print('Loading data')

        file = os.path.join(saving_path, 'variables', 'temp_dataset_15min.csv')
        df_15min_temp = pd.read_csv(file, delimiter=',', header=0, index_col="date")
        df_15min_temp.index = pd.to_datetime(df_15min_temp.index, infer_datetime_format=True)
        df_15min_temp.sort_index(inplace=True)
    print('______')
    return df_15min_temp


def prcp_momo_forest(path_gen: str, saving_path: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    path_gen: general path where the temperature files are located
    saving_path: saving path where the results are save

    Returns:
    Dataframe with the precipitation with dt = 15 mins
    -------

    """

    # check is file is already done
    print('______')
    print('Prcp treatment Foret Montmorency')

    if not os.path.isfile(os.path.join(saving_path, 'variables', 'prcp_dataset_15min.csv')):

        all_files_ott = sorted(glob.glob(os.path.join(path_gen, 'OTT', '*')))

        list_df_prcp_ott = []

        # open every monthly file and creat the 1 min dataframe.
        # The extreme are remove if value is 3 time sigma from the mean value

        for file in all_files_ott:

            dict_col = {"TIMESTAMP": "date",
                        "AccuPrecipUnfilt_DFAR": "precip_inst_pluvio_acc",
                        "BucketUnfilt_DFAR": "precip_tot_pluvio",
                        "PStatus_DFAR": 'status_prcp'}

            df_1month = pd.read_csv(file, delimiter=';', header=0)

            for col in df_1month.columns:
                if col not in dict_col.keys():
                    df_1month.drop(columns=col, axis=1, inplace=True)

            df_1month.rename(columns=dict_col, inplace=True)
            df_1month.set_index('date', drop=True, inplace=True)

            df_1month.index = pd.to_datetime(df_1month.index, infer_datetime_format=True)
            df_1month.sort_index(inplace=True)

            # if value is bigger than 1 than the bucket is full and the value after that number are nit accurate
            df_1month = df_1month[~(df_1month['status_prcp'] > 1)]
            df_1month.drop(columns='status_prcp', axis=1, inplace=True)
            df_1month.mask(df_1month['precip_inst_pluvio_acc'] == 999999, inplace=True)

            # remove unwanted pics that are more than 3 time the std deviation from the meab and negative values
            diff = (df_1month['precip_inst_pluvio_acc']
                    .fillna(method='bfill')
                    .diff())  # limit=12 trouve par essai/erreur
            diff_mask = (diff < 0) | (diff >= (diff.mean() + 3 * diff.std()))
            df_1month.loc[diff_mask, 'precip_inst_pluvio_acc'] = np.nan

            df_1month['precip_inst_pluvio_acc'].fillna(method='bfill')

            df_1month['precip_tot_pluvio'].fillna(method='bfill')

            list_df_prcp_ott.append(df_1month)

        all_files = sorted(glob.glob(os.path.join(path_gen, 'Geonor_UL', '*')))

        # open every monthly file and creat the 1 min dataframe for the geonor.
        # The extreme are remove if value is 3 time sigma from the mean value
        list_df_prcp_geonor = []
        for file in all_files:
            dict_col = {"TIMESTAMP": "date",
                        "Precip2_mm": "precip_tot_geonor",}

            df_1month = pd.read_csv(file, delimiter=';', header=0)

            for col in df_1month.columns:
                if col not in dict_col.keys():
                    df_1month.drop(columns=col, axis=1, inplace=True)

            df_1month.rename(columns=dict_col, inplace=True)
            df_1month.set_index('date', drop=True, inplace=True)

            df_1month.index = pd.to_datetime(df_1month.index, infer_datetime_format=True)
            df_1month.sort_index(inplace=True)


            list_df_prcp_geonor.append(df_1month)

        # traitement des donnée momo geonor
        df_prcp_1min_geonor = pd.concat(list_df_prcp_geonor)
        df_prcp_1min_geonor.sort_index(inplace=True)

        df_prcp_1min_ott = pd.concat(list_df_prcp_ott)
        df_prcp_1min_ott.sort_index(inplace=True)


        df_prcp_1min = pd.concat([df_prcp_1min_ott,df_prcp_1min_geonor],axis=1)
        df_prcp_1min.sort_index(inplace=True)
        df_prcp_1min = func_traitement(df_prcp_1min)

        resample_15m_dict = {
            'precip_inst_pluvio': 'sum',
            'precip_inst_geonor': 'sum'}

        df_15min_prcp = df_prcp_1min.resample('15T', label='left').agg(resample_15m_dict)
        df_15min_prcp.sort_index(inplace=True)
        df_15min_prcp['precip_tot_pluvio'] = df_15min_prcp['precip_inst_pluvio'].cumsum()
        df_15min_prcp['precip_tot_geonor'] = df_15min_prcp['precip_inst_geonor'].cumsum()

        df_15min_prcp.to_csv(os.path.join(saving_path, 'variables', 'prcp_dataset_15min.csv'))

    else:

        print('Loading data')

        file = os.path.join(saving_path, 'variables', 'prcp_dataset_15min.csv')
        df_15min_prcp = pd.read_csv(file, delimiter=',', header=0, index_col="date")
        df_15min_prcp.index = pd.to_datetime(df_15min_prcp.index, infer_datetime_format=True)
        df_15min_prcp.sort_index(inplace=True)
    print('______')
    return df_15min_prcp



def func_traitement(df_prcp):

    list_var_acc_tot = ['precip_tot_geonor','precip_tot_pluvio']
    dict_precip = {'precip_tot_pluvio': 'precip_inst_pluvio',
                   'precip_tot_geonor': 'precip_inst_geonor'}

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
        # plt.plot(df_prcp.index, df_prcp[cummu])

        # Raccrochage des données





        # plt.plot(df_prcp.index, df_prcp[cummu].diff())
        df_prcp[inst] = df_prcp[cummu].diff()
        df_prcp.loc[(df_prcp[inst]< -2) | (df_prcp[inst] > 10),inst] = 0
        df_prcp[cummu] = df_prcp[inst].cumsum()

        diff = (df_prcp[cummu]
                .fillna(method='bfill', limit=12)
                .diff())
        # plt.plot(df_prcp.index, df_prcp[inst].cumsum())
        # plt.show()



        # diff_mask = (diff < 0)
        #
        # pdt = pd.to_timedelta("00:15:00")
        #
        # window_end = df_prcp.loc[diff_mask, cummu].index
        # window_start = window_end - 10 * pdt  # testé pour 10*pdt, quantité à généraliser
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

        # plt.plot(df_prcp.index, df_prcp[cummu])

        # Retrait des "jumps"
        # jumps = diff.copy()

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

            xt = df_prcp[inst].index
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



def wind_momo_forest(path_gen: str, saving_path: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    path_gen: general path where the winds files are located
    saving_path: saving path where the results are save

    Returns:
    Dataframe with the winds with dt = 15 mins
    -------

    """

    print('______')
    print('Wind treatment Foret Montmorency')

    if not os.path.isfile(os.path.join(saving_path, 'variables', 'wind_dataset_15min.csv')):

        all_files = sorted(glob.glob(os.path.join(path_gen, 'Young', '*')))

        list_df_wind = []

        # open every monthly file and creat the 1 min dataframe.
        # The extreme are remove if value is 3 time sigma from the mean value

        for file in all_files:
            df_1month = pd.read_csv(file, delimiter=';', header=0)

            dict_col = {"TIMESTAMP": "date",
                        "WS_ms_S_WVT": 'vitesse_vent_moy_2m',
                        "WindDir_D1_WVT": 'dir_vent_moy_2m'}

            df_1month.rename(columns=dict_col, inplace=True)
            df_1month.set_index('date', drop=True, inplace=True)

            df_1month.index = pd.to_datetime(df_1month.index, infer_datetime_format=True)
            df_1month.sort_index(inplace=True)

            df_1month.drop(columns='RECORD', axis=1, inplace=True)
            df_1month.drop(columns='WindDir_SD1_WVT', axis=1, inplace=True)

            # erase negative and over 360 deg value.
            df_1month.loc[
                (df_1month['dir_vent_moy_2m'] < 0) | (df_1month['dir_vent_moy_2m'] > 360), 'dir_vent_moy_2m'] = np.nan
            df_1month.loc[
                (df_1month['vitesse_vent_moy_2m'] < 0) | (df_1month['vitesse_vent_moy_2m'] > 150/3.6), 'dir_vent_moy_2m'] = np.nan
            list_df_wind.append(df_1month)

        df_wind_1min = pd.concat(list_df_wind)
        df_wind_1min.sort_index(inplace=True)
        resample_15m_dict = {
            'dir_vent_moy_2m': 'mean',
            'vitesse_vent_moy_2m': 'mean'}
        df_15min_wind = df_wind_1min.resample('15T', label='left').agg(resample_15m_dict)

        df_15min_wind.to_csv(os.path.join(saving_path, 'variables', 'wind_dataset_15min.csv'))

    else:

        print('Loading data')

        file = os.path.join(saving_path, 'variables', 'wind_dataset_15min.csv')
        df_15min_wind = pd.read_csv(file, delimiter=',', header=0, index_col="date")
        df_15min_wind.index = pd.to_datetime(df_15min_wind.index, infer_datetime_format=True)
        df_15min_wind.sort_index(inplace=True)
    print('______')
    return df_15min_wind


def parsivel_momo_forest(path_gen: str, saving_path: str) -> pd.DataFrame:
    """
    This function treat the parsivel data to make them in the same format as the HQ station.
    I decide to simply take every prcp type at each 15min instead of interpolating to get the weather code.

    I'm gonna check if it's a good idea, i can always rework it later


    Parameters
    ----------
    path_gen: path where the parsivel files are located
    saving_path: saving path where the results are save

    Returns:
    Dataframe with the phase of the precipitation with dt = 15 mins
    -------
    """
    print('______')
    print('Parsivel treatment Foret Montmorency')

    if not os.path.isfile(os.path.join(saving_path, 'variables', 'parsivel_dataset_15min.csv')):

        all_files = sorted(glob.glob(os.path.join(path_gen, 'Parsivel', '*')))

        list_df_parsivel = []

        # open every monthly file and creat the 1 min dataframe.
        # The extreme are remove if value is 3 time sigma from the mean value

        for file in all_files:
            dict_col = {"TIMESTAMP": "date",
                        "WeatherCode": 'type_precip',
                        }

            df_1month = pd.read_csv(file, delimiter=';', header=0)

            for col in df_1month.columns:
                if col not in dict_col.keys():
                    df_1month.drop(columns=col, axis=1, inplace=True)

            df_1month.rename(columns=dict_col, inplace=True)
            df_1month.set_index('date', drop=True, inplace=True)

            df_1month.index = pd.to_datetime(df_1month.index, infer_datetime_format=True)
            df_1month.sort_index(inplace=True)

            list_df_parsivel.append(df_1month)

        df_parsivel_1min = pd.concat(list_df_parsivel)
        df_parsivel_1min.sort_index(inplace=True)
        df_parsivel_1min.loc[(df_parsivel_1min['type_precip'] > 100), 'type_precip'] = np.nan

        # The mode is used to get the most recurent code during the 15 min interval
        resample_15m_dict = {
            'type_precip': mode}
        df_15min_parsivel = df_parsivel_1min.resample('15T', label='left').agg(resample_15m_dict)

        df_15min_parsivel = standard_prcp_type(df_15min_parsivel)


        df_15min_parsivel.to_csv(os.path.join(saving_path, 'variables', 'parsivel_dataset_15min.csv'))

    else:
        print('Loading data')
        file = os.path.join(saving_path, 'variables', 'parsivel_dataset_15min.csv')
        df_15min_parsivel = pd.read_csv(file, delimiter=',', header=0, index_col="date")
        df_15min_parsivel.index = pd.to_datetime(df_15min_parsivel.index, infer_datetime_format=True)
        df_15min_parsivel.sort_index(inplace=True)

    print('______')

    return df_15min_parsivel


def standard_prcp_type(df: pd.DataFrame) -> pd.DataFrame:
    # transform other type of precip into the standrar type

    list_part_type = []

    for code in df['type_precip']:

        if code in [51, 52, 53, 57, 58, 61, 62, 63]:
            # tranform to rain
            list_part_type.append(60)
        elif code in [67]:
            # tranform to freezing rain
            list_part_type.append(67)
        elif code in [68]:
            # tranform to rain or drizzel and snow
            list_part_type.append(69)
        elif code in [71, 72, 73, 77, 87, 88, 89]:
            # tranform to snow
            list_part_type.append(70)
        elif code in [0]:
            list_part_type.append(0)
        elif np.isnan(code):
            list_part_type.append(np.nan)

    df['type_precip'] = list_part_type

    return df

def pdt_15min_to_1h(df: pd.DataFrame, save_path: str):
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

    resample_1h_op = {
        'humidite_air': 'mean',
        'temp_moy': 'mean',
        'temp_max': 'max',
        'temp_min': 'min',
        'precip_inst_pluvio': 'sum',
        'precip_inst_geonor': 'sum',
        'dir_vent_moy_2m': 'mean',
        'vitesse_vent_moy_2m': 'mean'}

    print('Resampling...')


    resamp, frac_list, frac_list_qty = phase_frac(df)

    df_1h = df.resample('1h', label='left').agg(resample_1h_op)
    df_1h.loc[df_1h['precip_inst_pluvio'] < 0.2, ['precip_inst_pluvio']] = 0
    df_1h.loc[df_1h['precip_inst_pluvio'] > 110, ['precip_inst_pluvio']] = np.nan


    df_1h = pd.concat([df_1h, resamp[frac_list + frac_list_qty]], axis=1)


    df_1h['precip_tot_geonor'] = df_1h['precip_inst_geonor'].cumsum()

    # undercatchment
    lidx = df_1h['precip_tot_geonor'].last_valid_index()
    unprcp = df_1h.loc[lidx, 'precip_tot_geonor']

    mask = df_1h['precip_inst_geonor'] > 0

    for frac in frac_list:
        df_1h.loc[mask, frac] = df_1h.loc[mask, frac] / df_1h.loc[mask, 'precip_inst_geonor']

    df_1h = sous_captation_df_2m(df_1h)

    for frac in frac_list:
        df_1h.loc[mask, frac] = df_1h.loc[mask, frac] * df_1h.loc[mask, 'precip_inst_geonor']

    toprcp = df_1h.loc[lidx, 'precip_tot_geonor']

    print(f'MOMO Diff touch-untouch: {toprcp - unprcp:.2f} mm')
    print(f'MOMO Diff touch-untouch: {((toprcp - unprcp) / unprcp) * 100:.2f} %')


    print('Sauvegarde de la base de donnée horaire...')
    df_1h.to_csv(os.path.join(save_path, 'dataset_1h.csv'))

def sous_captation_df_2m(df_capt: pd.DataFrame) -> pd.DataFrame:

    times = df_capt.index
    for i, time in enumerate(times):
        row = df_capt.loc[time]
        row_temp = row['temp_moy']
        row_wind = row['vitesse_vent_moy_2m']
        row_prcp = row['precip_inst_pluvio']

        # min of prcp and temp not none
        if row_prcp >= 0.22 and not np.isnan(row_temp) and row_temp <= 5:
            # solid

            # CTE from Kochendorfer et all. 2017 table 2
            a, b, c = 0.0348, 1.366, 0.779

            # wind threshold
            if row_wind >= 7.2:
                row_wind = 7.2
            else:
                row_wind = row_wind

            CE_row = catch_effic_eq3(row_wind, row_temp, a, b, c)
            if CE_row > 1:
                df_capt.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor']
            else:
                df_capt.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row



        else:

            df_capt.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor']


    df_capt['precip_tot_geonor'] = df_capt['precip_inst_geonor'].cumsum()

    return df_capt
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
def phase_frac(df_phase):
    frac_list = []
    frac_list_qty = []
    phase_list = (df_phase['type_precip'].unique())

    for phase in phase_list:

        if np.isnan(phase):
            df_phase.loc[df_phase['type_precip'].isna(),
                         f'#{phase}_1h'] = 1
            frac_list_qty.append(f'#{phase}_1h')

        else:
            phase_col = 'f_' + str(int(phase))
            df_phase[phase_col] = np.nan
            frac_list.append(phase_col)

            df_phase.loc[df_phase['type_precip'] == phase,
                         phase_col] = df_phase.loc[df_phase['type_precip'] == phase,
                                                   'precip_inst_geonor']

    resamp_phase = df_phase[frac_list + frac_list_qty].resample('1h', label='left').sum()

    resamp_phase['tot'] = resamp_phase[frac_list].sum(axis=1)

    # resamp_phase.drop(columns=['tot'], inplace=True)

    resamp_phase.fillna(0, inplace=True)

    return resamp_phase, frac_list, frac_list_qty

def fz_correction_temp(df:pd.DataFrame)->pd.DataFrame:
    """
    correction type aprticule if rain with temp < 0 -> FZ
    Parameters
    ----------
    df

    Returns
    -------

    """

    diff_mask = (df['type_precip'] == 60) & (df['temp_moy'] < 0)
    df.loc[diff_mask, 'type_precip'] = 67

    return df
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
