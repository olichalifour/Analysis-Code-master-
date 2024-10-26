"""
Author: Olivier Chalifour
Organisation: UQAM
Date: 20 October 2021

---


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# changer pour une correction au heurs
def undercatchement_hq(data_path: str):
    all_files = [fn for fn in glob.glob(os.path.join(data_path, '*.csv'))
                 if os.path.basename(fn).endswith(tuple(['1h.csv']))]

    for filename in all_files:

        print('Traitement de la sous-captation du fichier ' + filename)
        df = pd.read_csv(filename, parse_dates=['date'])
        df = df.set_index('date')
        df.sort_index(inplace=True)

        list_corrected_dataframe = []
        j = 0

        frac_list_30min = ['f_0', 'f_60', 'f_67', 'f_69', 'f_70', 'f_nan']

        list_subdf_acc = []
        for name, group in df.groupby('filename'):

            group['precip_tot_pluvio'] = group['precip_inst_pluvio'].cumsum()
            try:
                lidx = group['precip_tot_pluvio'].last_valid_index()
                # untouchprcp = group.loc[lidx, 'precip_tot_pluvio']
                untouchprcp = np.sum(group['precip_inst_pluvio'])
            except:
                pass

            for frac in frac_list_30min:
                mask = group['precip_inst_pluvio'] > 0
                group.loc[mask, frac] = group.loc[mask, frac] / group.loc[mask, 'precip_inst_pluvio']

            group = sous_captation_df_2m(group)

            for frac in frac_list_30min:
                mask = group['precip_inst_pluvio'] > 0
                group.loc[mask, frac] = group.loc[mask, frac] * group.loc[mask, 'precip_inst_pluvio']

            list_subdf_acc.append(group.loc[:, ['precip_inst_pluvio', 'filename']])

            list_corrected_dataframe.append(group)

            try:
                # touchprcp = list_corrected_dataframe[j].loc[lidx, 'precip_tot_pluvio']
                touchprcp = np.sum(list_corrected_dataframe[j]['precip_inst_pluvio'])
                print(f'{name}: Diff touch-untouch: {touchprcp - untouchprcp:.2f} mm')
                print(f'{name}: Diff touch-untouch: {((touchprcp - untouchprcp) / untouchprcp) * 100:.2f} %')
            except:
                print(f'{name}: Diff touch-untouch: nan by idx (no valid last idx)')

            j += 1

        df_cor = pd.concat(list_corrected_dataframe)

        print('Sauvegarde de la base de données... \n')

        df_cor.to_csv(all_files[0])

    df_undecatch_corre = pd.concat(list_subdf_acc)
    df_undecatch_corre.to_csv('/Users/olivier1/Documents/GitHub/data_format-master/Data.nosync/df_acc_undercatch.csv')




def sous_captation_df_2m(df_capt: pd.DataFrame) -> pd.DataFrame:
    times = df_capt.index
    for i, time in enumerate(times):
        row = df_capt.loc[time]
        row_temp = row['temp_moy']

        row_wind = row['vitesse_vent_moy_2_5m'] / 3.6
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
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']
            else:
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio'] / CE_row

        else:

            df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']

    df_capt['precip_tot_pluvio'] = df_capt['precip_inst_pluvio'].cumsum()

    return df_capt


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
        row_wind = row['vitesse_vent_moy_2_5m'] / 3.6
        row_prcp = row['precip_inst_pluvio']

        # min of prcp and temp not none
        # and row_temp >= -20
        if row_prcp >= 0.22 and not np.isnan(row_temp) and row_temp <= 5:

            # CTE from Kochendorfer et all. 2017 table 2
            a, b, c = 0.0348, 1.366, 0.779

            # wind threshold
            if row_wind >= 7.2:
                row_wind = 7.2
            else:
                row_wind = row_wind
            CE_row = catch_effic_eq3(row_wind, row_temp, a, b, c)
            # if CE_row > 1:
            #     print(CE_row, row_type, row_temp, row_wind, row_prcp)
            if CE_row > 1:
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']
            else:
                df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio'] / CE_row

        else:

            df_capt.loc[time, 'precip_inst_pluvio'] = row['precip_inst_pluvio']

    df_capt['precip_tot_pluvio'] = df_capt['precip_inst_pluvio'].cumsum()

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


def catch_effic_eq4(wind, a, b, c) -> float:
    """
    Catch efficientcy
    Kochendorfer et all. 2017 eq. (4)
    Parameters
    ----------
    wind: wind speed
    a: parameter from the article Kochendorfer et all. 2017
    b: parameter from the article Kochendorfer et all. 2017
    c: parameter from the article Kochendorfer et all. 2017

    Returns
    -------

    """
    CE = a * np.e ** (-b * wind) + c

    return CE
