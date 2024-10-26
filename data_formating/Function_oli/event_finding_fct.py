# -*- coding: utf-8 -*-
"""
Created by: Olivier Chalifour
Date: 12-09-2022

"""


import pandas as pd
import numpy as np
import os


def event_finding_fct(path_df:str):

    df_1h = pd.read_csv(os.path.join(path_df, 'dataset_1h.csv'), header=0, index_col='date')
    df_summary = pd.DataFrame(columns=['nom', 'begin','end'])

    for station, subdf in df_1h.groupby('filename'):

        subdf.index = pd.to_datetime(subdf.index, infer_datetime_format=True)
        subdf.sort_index(inplace=True)

        # subdf.mask(subdf['precip_inst_pluvio'] < 0.1,0)
        accu_prcp = subdf['precip_inst_pluvio'].values
        list_idx_no_nul = np.nonzero(accu_prcp)[0]


        non_nul_df = subdf.iloc[list_idx_no_nul]
        non_nul_df_index = non_nul_df.index

        time_delta_6h = pd.Timedelta(hours=6)
        list_event = [[non_nul_df_index[0]]]
        i = 0
        # divide the time in different event
        for idx in range(len(non_nul_df_index)-1):
            time_x = non_nul_df_index[idx]
            time_x_1 = non_nul_df_index[idx+1]

            if time_x_1 >= time_x+time_delta_6h:
                list_event[i].append(time_x_1)
                i+=1
                list_event.append([])
            elif time_x_1 < time_x+time_delta_6h:
                list_event[i].append(time_x_1)

        for event in list_event:
            if len(event) > 3:
                subdf_event = subdf.loc[event]
                phase_fr = subdf_event['#67_1h'].values
                # get if there is freezing rain during the event
                if np.count_nonzero(phase_fr == 4) > 1:
                    begin = event[0]
                    end = event[-1]
                    dict_update = {'nom': station,'begin':begin,'end':end}
                    df_summary = df_summary.append(dict_update, ignore_index=True,)

    df_summary.sort_values('begin',inplace=True)
    df_summary.to_excel(f'{path_df}/summary_date_event.xlsx')






