
# -*- coding: utf-8 -*-
"""
@author: Olivier
"""

import pandas as pd
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def dfholes(path_df_15min: str ) -> List:
    """
    Parameters
    ----------
    DF: pandas dataframe avec les datas et une colonne de temps
    name_datetime: le nom de la colonne de temps
    step_datetime: le pas de temps utilise

    Returns
    -------
    List des pas de temps qui manque
    """
    # Open dataframe

    df_15min = pd.read_csv(os.path.join(path_df_15min, 'dataset_15min_formated.csv'),header=0,index_col = 'date')
    # df_15min = pd.read_csv(os.path.join(path_df_15min, 'dataset_1h.csv'), header=0, index_col='date')

    dict_sta = {}
    list_stat = []

    for station, subdf in df_15min.groupby('filename'):

        print('---------')
        print(f'Holes Analysis for {station}')
        print('---------')

        subdf.index = pd.to_datetime(subdf.index, infer_datetime_format=True)
        subdf.sort_index(inplace=True)

        # mask data for the type prcp under zero in the dataframe
        # subdf.mask(subdf["type_precip"] < 0, inplace=True)

        # find Na hole
        na_hole = find_na_hole(subdf)

        dict_sta[station] = na_hole
        dict_sta[f'{station}_index'] = subdf.index
        # list_stat.append(station)
        # print(subdf.loc[na_hole])


        # plot_hole_na(na_hole, station)
        # summary = summary_na_hole(subdf,station)
        # print(summary)


        # find temporal hole
        # hole_missing_month = find_temporal_hole(subdf,station)

    plot_hole_na_all(dict_sta, list_stat)


def find_temporal_hole(subdf:pd.DataFrame,station:str)-> np.array:
    """

    Parameters
    ----------
    subdf:
    station:

    Returns
    -------

    """
    # if station == 'MOUCHA_M':
    #     subdf=subdf.loc['2021']

    begin = pd.to_datetime('2020-10')
    end = pd.to_datetime('2022-06')

    # get list of dates
    ldates = subdf.index

    range_date = pd.date_range(start=ldates[0], end=ldates[-1], freq='15 T')

    diff = range_date.difference(ldates)


    # date que j'ai de besoin
    set_date_not_need_month = {6, 7, 8, 9}

    # print(f'nb temporal hole:{len(diff)}')

    year_month_missing = list(set(diff.year))
    diff_month_needed = list(set(diff.month) - set_date_not_need_month)
    print(f"{station}:")


    # get missing month at the begining of the time range
    if ldates[0] > begin:
        range_month = np.arange(ldates[0].month, begin.month+1)
        for month_before in range_month:
            print(f"Month missing {begin.year}-{month_before}")

        # print(diff_month_needed)
    # get missing month in the time range
    for i in diff_month_needed:
        print(f"Month missing {year_month_missing[0]}-{i}")

    # get missing month at the end of the time range
    if ldates[-1] < end:
        range_month = np.arange(ldates[-1].month, end.month)
        for month_after in range_month:
            print(f"Month missing {end.year}-{month_after}")

    print('\n')

    return diff

def find_na_hole(subdf:pd.DataFrame)->np.array:

    list_nan_type_prcp = subdf[subdf["type_precip"].isna()].index.tolist()

    return list_nan_type_prcp



def summary_na_hole(subdf:pd.DataFrame,station:str)->pd.DataFrame:


    mask = subdf.type_precip.isna()
    d = subdf.index.to_series()[mask].groupby((~mask).cumsum()[mask]).agg(['first', 'size'])
    d.rename(columns=dict(size='num of contig null', first='Start_Date')).reset_index(drop=True)

    # get the biggest NA hole
    # biggest_hole = d.loc[d['size'].idxmax()]

    # print(f'{station}')
    # print(f'Hole start: {biggest_hole["first"]}')
    # print(f'for {biggest_hole["size"]} * 15 mins')
    # print(f'Or for {biggest_hole["size"] / 4:.2f} hours ')
    # print(f'Or for {biggest_hole["size"] / (4 * 24):.2f} days\n')

    # find n largest hole
    n = 3
    # print(d['size'].nlargest(n))

    # find hole with more than 10 step
    # biggest = d[d['size'] > 10]
    # print(biggest)


    return d

def plot_hole_na(na_time:np.array,station:str):



    begin = pd.to_datetime('2020-10')
    end = pd.to_datetime('2022-06')

    na_one = np.ones(len(na_time))

    fig = plt.figure(facecolor='white', dpi=200, figsize=(7, 5))
    # figsize=figsize
    # Set projection defined by the cartopy object
    ax = fig.add_subplot()
    df_na = pd.DataFrame(na_one,na_time)
    # df_na.index = pd.to_datetime(df_na.index, infer_datetime_format=True)
    # subdf.sort_index(inplace=True)

    df_na = df_na.resample('60T').sum()




    ax.axvspan(begin,end,facecolor='grey',alpha=0.3)
    ax.axvline(x=begin,c='k',linewidth=1)
    ax.axvline(x=end,c='k',linewidth=1)
    ax.bar(df_na.index, df_na[0])
    ax.set_title(station)

    locator_minor = mdates.HourLocator(interval=24 * 30)
    locator_major = mdates.HourLocator(interval=24 * 365)
    ax.set_xlabel('Time [UTC]', fontsize=10)

    plt.yticks(np.arange(0,5,1), ['0 min','15min','30 min','45 min','60 min'])
    ax.xaxis.set_major_locator(locator_major)
    ax.xaxis.set_minor_locator(locator_minor)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='x', which='minor', labelsize=8)
    fig.savefig(f'/Users/olivier1/Documents/GitHub/data_format-master/figures/hole_na_bar_{station}.png', dpi=200, format='png',
                bbox_inches='tight', )  # Most backends support png, pdf,
    # plt.show()
    plt.close()


def plot_hole_na_all(na_stat_dict:dict,list_station:list):
    begin_xaxis = pd.to_datetime('2019-04')
    end_xaxis = pd.to_datetime('2022-10')

    begin = pd.to_datetime('2020-10')
    end = pd.to_datetime('2022-06')

    fig = plt.figure(facecolor='white', dpi=200, figsize=(9, 7))
    # figsize=figsize
    # Set projection defined by the cartopy object
    ax = fig.add_subplot()

    for i, stat in enumerate(list_station):
        na_time = na_stat_dict[stat]
        idx_time = na_stat_dict[f'{stat}_index']

        na_one = np.ones(len(na_time))*i
        idx_one = np.ones(len(idx_time)) * i
        if i ==0:
            ax.scatter(idx_time, idx_one, label='Data', c='k')
            ax.scatter(na_time, na_one, label='hole', c='r')
        else:
            ax.scatter(idx_time, idx_one, c='k')
            ax.scatter(na_time, na_one, c='r')


    ax.axvspan(begin, end, facecolor='grey', alpha=0.3)
    ax.axvline(x=begin, c='k', linewidth=1)
    ax.axvline(x=end, c='k', linewidth=1)
    locator_minor = mdates.MonthLocator(interval=1)
    locator_major = mdates.YearLocator()
    ax.set_xlabel('Time [UTC]', fontsize=10)
    ax.xaxis.set_major_locator(locator_major)
    ax.xaxis.set_minor_locator(locator_minor)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m\n%Y"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='x', which='minor', labelsize=8)
    ax.legend(ncol=1,loc='lower center',bbox_to_anchor=(1.06, 0),fontsize=10)
    plt.yticks(np.arange(0, len(list_station), 1), list_station)
    plt.tight_layout()
    ax.set_xlim(begin_xaxis, end_xaxis)
    fig.savefig('/Users/olivier1/Documents/GitHub/data_format-master/figures/hole_na_stats.png', dpi=200, format='png',
                bbox_inches='tight', )  # Most backends support png, pdf,
    # plt.show()
    plt.close()

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
def summary_file_1h(path_file_1h:str):
    begin = pd.to_datetime('2020-10')
    end = pd.to_datetime('2022-06')


    df_1h = pd.read_csv(os.path.join(path_file_1h, 'dataset_1h.csv'), header=0, index_col='date')

    df_summary = pd.DataFrame(columns=['nom','% 0','% 60','% 67','% 69','% nan','nb_tot_points'])


    for station, subdf in df_1h.groupby('filename'):
        dict_update = dict.fromkeys(['nom', '% 0', '% 60', '% 67', '% 69', '% nan', 'nb_tot_points'])
        dict_update['nom'] = station
        dict_update['nom_stat_didro'] = station
        dict_update['nom_stat_meteo'] = st_dict[station]
        dict_update['Altitude_stat_disdro'] = elev_dict[station]
        dict_update['Altitude_stat_meteo'] = elev_dict_stat_meteo[st_dict[station]]

        print('---------')
        print(f'Holes Analysis for {station}')
        print('---------')


        subdf.index = pd.to_datetime(subdf.index, infer_datetime_format=True)
        subdf.sort_index(inplace=True)

        mask = (subdf.index >= begin) & (subdf.index <= end)
        subdf = subdf.loc[mask]

        frac_list = ['#nan_1h','#0_1h','#60_1h', '#67_1h', '#69_1h', '#70_1h']
        # nb_pt = len(subdf_15min.index)
        nb_pt = subdf[frac_list].sum(axis=1).sum(axis=0)

        sum_frac_sum = 0

        for frac in frac_list:
            sum_frac = subdf[frac].sum()
            sum_frac_sum += (sum_frac/nb_pt)*100

            dict_update[f'% {frac.strip("#_1h")}'] = (sum_frac/nb_pt)*100

            print(f'Fraction de phase {frac.strip("#_1h")} : {(sum_frac/nb_pt)*100:.2f} %')

        dict_update[f'nb_tot_points'] = nb_pt

        df_summary = df_summary.append(dict_update,ignore_index=True)

    df_summary.to_excel(f'{path_file_1h}/summary_data_phase_stat.xlsx')













