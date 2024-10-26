# -*- coding: utf-8 -*-
"""
Created by: Olivier Chalifour
Date: 20-09-2022

"""
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import glob
import os
import warnings
import matplotlib.pyplot as plt
from Functions.formating_functions import NAF_SEG
from scipy import stats

begin, end = '2021-10', '2022-06'
time_range_hours = pd.date_range(begin, end.replace("7", "6"), freq='H', closed='right')[:-1]
list_stat_prob = ['GAREMANG', 'PIPMUA_G']

def stat_analysis_r(saving_path):
    all_files = [fn for fn in glob.glob(os.path.join(saving_path, '*.csv'))
                 if os.path.basename(fn).endswith(tuple(['1h.csv']))]

    all_files_15min = [fn for fn in glob.glob(os.path.join(saving_path, '*.csv'))
                 if os.path.basename(fn).endswith(tuple(['15min_formated.csv']))]



    for filename,filename_15min in zip(all_files,all_files_15min):

        print('Traitement du fichier ' + filename + '... \n')

        df_1h = pd.read_csv(filename, parse_dates=['date'])
        df_1h = df_1h.set_index('date')

        df_15min = pd.read_csv(filename_15min, parse_dates=['date'])
        df_15min = df_15min.set_index('date')

        dict_stat = {}

        for stat,subdf in df_1h.groupby('filename'):

            subdf = subdf.loc[begin:end]

            if len(subdf.index) != 0:
                if (subdf['#nan_1h'].sum() / (4 * len(time_range_hours)) * 100 < 10) \
                        and (stat not in list_stat_prob) \
                        and (subdf['precip_inst_pluvio'].sum() > 0):

                    subdf_15min = df_15min.loc[df_15min['filename'] == stat]

                    subdf_15min = subdf_15min.loc[begin:end]


                    nb_disdro = np.sum(subdf_15min['precip_inst_disdro']/subdf_15min['precip_inst_disdro'])
                    nb_pluvio = np.sum(subdf_15min['precip_inst_pluvio']/subdf_15min['precip_inst_pluvio'])

                    print(stat)
                    print(f'Disdro : {nb_disdro}')
                    print(f'Pluvio : {nb_pluvio}')
                    print(f'Rapport: {((nb_disdro-nb_pluvio)/nb_disdro)*100}')

                    dict_stat[stat] =  {'pluvio':nb_pluvio,'disdro':nb_disdro}


        plot_analysis(dict_stat)


def plot_analysis(dict_stat):
    fig = plt.figure(facecolor='white', dpi=200, figsize=(15, 4))
    ax = plt.axes()
    width = 0.4
    station = dict_stat.keys()
    x = np.arange(len([i for i in station])) * 2


    lonlat_path = '/Users/olivier1/Documents/GitHub/data_format-master/Data.nosync/Disdrometres_coordonnées.csv'
    df_disdro = pd.read_csv(lonlat_path, header=0)
    df_disdro.set_index('Name', inplace=True)
    station_ordered=df_disdro.loc[station].sort_values('Y').index

    for i,stat in enumerate(station_ordered):
        y_pluvio=dict_stat[stat]['disdro']
        y_disdro=dict_stat[stat]['pluvio']
        if i == 0 :
            ax.scatter(x[i], y_pluvio, label='Disdro', s=30, c='b', zorder=9999)
            ax.scatter(x[i], y_disdro, label='Pluvio', s=30, c='r', zorder=9999)
        else:
            ax.scatter(x[i], y_pluvio, s=30, c='b', zorder=9999)
            ax.scatter(x[i], y_disdro, s=30, c='r', zorder=9999)
        ax.set_ylim(1000, 5000)

        if y_pluvio > y_disdro:

            rect = mpatches.Rectangle((x[i]-width, y_disdro), 2*width, y_pluvio-y_disdro,
                                      # fill=False,
                                      facecolor='grey',alpha=0.5,
                                      edgecolor='k')
            plt.gca().add_patch(rect)

            ax.text(x[i],
                    (y_pluvio+y_disdro)/2,
                    f"{((y_pluvio-y_disdro)/y_disdro)*100:.0f} %",
                    zorder=9999,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)


        else:

            rect = mpatches.Rectangle((x[i] - width, y_pluvio), 2*width, y_disdro - y_pluvio,
                                      # fill=False,
                                      facecolor='grey', alpha=0.5,
                                      edgecolor='k')
            ax.text(x[i],
                    (y_pluvio + y_disdro) / 2,
                    f"{((y_pluvio - y_disdro) / y_disdro) * 100:.0f} %",
                    zorder=9999,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)
            plt.gca().add_patch(rect)


    time_text = f'{time_range_hours[0].strftime("%Y-%m-%d")} to {time_range_hours[-1].strftime("%Y-%m-%d")}'

    ax.annotate(f'{time_text}', xy=(0.78, 1.01), xycoords='axes fraction', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(station_ordered, rotation=45, ha='right')
    ax.set_ylabel('Nb of observation [-]', fontsize=12)
    ax.set_xlabel('Stations', fontsize=12)

    ax.legend()

    plt.savefig(f'/Users/olivier1/Documents/GitHub/data_format-master/figures/analysis_fig_stat_dvsp_{time_range_hours[0].strftime("%Y")}.png',dpi=200, format='png', bbox_inches='tight')
    # plt.show()





