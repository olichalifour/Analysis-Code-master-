import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as smt
import scipy.stats as statis
import sklearn.metrics as metrics
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
project_path =  os.path.abspath(os.path.join(__file__ ,"../.."))

savingpath_data_momo = os.path.join(project_path, r"Data.nosync/site_neige/Full_dataset")


all_files = [fn for fn in glob.glob(os.path.join(savingpath_data_momo, '*.csv'))
                 if os.path.basename(fn).endswith(tuple(['15min_total_formated.csv']))]

resample_1h_op = {
        'humidite_air': 'mean',
        'temp_moy': 'mean',
        'temp_max': 'max',
        'temp_min': 'min',
        'precip_inst_pluvio': 'sum',
        'precip_inst_geonor': 'sum',
        'dir_vent_moy_2m': 'mean',
        'vitesse_vent_moy_2m': 'mean'}
def cas_1(df_fct_1:pd.DataFrame):
    """
    E.q.3 avec temperature threshold
    Parameters
    ----------
    df

    Returns
    -------

    """

    times = df_fct_1.index
    for i, time in enumerate(times):
        row = df_fct_1.loc[time]
        row_temp = row['temp_moy']
        # deja en m/s à momo
        row_wind = row['vitesse_vent_moy_2m']
        row_prcp = row['precip_inst_geonor']

        # min of prcp and temp not none
        if row_prcp > 0.2 and not np.isnan(row_temp) and row_temp <= 5 :

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
            df_fct_1.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row

        else:

            df_fct_1.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor']

    df_fct_1['precip_tot_geonor'] = df_fct_1['precip_inst_geonor'].cumsum()
    # plt.plot(df.index, df['precip_tot_geonor'], label='cas_1')

    return df_fct_1


def cas_2(df_fct_2:pd.DataFrame):
    """
    E.q.4 avec temperature threshold pour la phase
    Parameters
    ----------
    df

    Returns
    -------

    """
    times = df_fct_2.index
    for i, time in enumerate(times):
        row = df_fct_2.loc[time]
        row_temp = row['temp_moy']
        # deja en m/s a momo
        row_wind = row['vitesse_vent_moy_2m']
        row_prcp = row['precip_inst_geonor']


        # min of prcp and temp not none
        if row_prcp > 0.2:
            # solid
            if row_temp < -2:
                # CTE from Kochendorfer et all. 2017 table 2
                a, b, c = 0.728,0.230,0.336
                if row_wind >= 7.2:
                    row_wind = 7.2
                else:
                    row_wind = row_wind
                CE_row = catch_effic_eq4(row_wind, a, b, c)
                # if CE_row > 1:
                #     print(CE_row, row_type, row_temp, row_wind, row_prcp)
                df_fct_2.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row

            # mix
            elif row_temp >= -2 and row_temp <= 2:
                a, b, c = 0.668, 0.132, 0.339
                if row_wind >= 7.2:
                    row_wind = 7.2
                else:
                    row_wind = row_wind
                CE_row = catch_effic_eq4(row_wind, a, b, c)
                # if CE_row > 1:
                #     print(CE_row, row_type, row_temp, row_wind, row_prcp)

                df_fct_2.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row


            else:
                df_fct_2.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor']

        else:

            df_fct_2.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor']

    df_fct_2['precip_tot_geonor'] = df_fct_2['precip_inst_geonor'].cumsum()
    # plt.plot(df.index, df['precip_tot_geonor'], label='cas_2')
    return df_fct_2


def cas_3(df_fct_3:pd.DataFrame):
    """
   E.q.4 avec phase threshold pour la phase
   Parameters
   ----------
   df

   Returns
   -------

   """
    times = df_fct_3.index
    for i, time in enumerate(times):
        row = df_fct_3.loc[time]
        row_temp = row['temp_moy']
        # deja en m/s a momo
        row_wind = row['vitesse_vent_moy_2m']
        row_prcp = row['precip_inst_geonor']
        row_type = row['type_precip']

        # min of prcp and temp not none
        if row_prcp >= 0.05:
            # solid
            if row_type == 70:
                # CTE from Kochendorfer et all. 2017 table 2
                a, b, c = 0.728, 0.230, 0.336
                # wind threshold
                if row_wind >= 7.2:
                    row_wind = 7.2
                else:
                    row_wind = row_wind
                CE_row = catch_effic_eq4(row_wind, a, b, c)
                # if CE_row > 1:
                #     print(CE_row, row_type, row_temp, row_wind, row_prcp)
                df_fct_3.loc[time, 'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row
            # mix
            elif row_type == 69:
                a, b, c = 0.668, 0.132, 0.339
                # wind threshold
                if row_wind >= 7.2:
                    row_wind = 7.2
                else:
                    row_wind = row_wind
                CE_row = catch_effic_eq4(row_wind, a, b, c)

                # if CE_row > 1:
                #     print(CE_row, row_type, row_temp, row_wind, row_prcp)
                df_fct_3.loc[time,'precip_inst_geonor'] = row['precip_inst_geonor'] / CE_row

            else:
                df_fct_3.loc[time,'precip_inst_geonor'] = row['precip_inst_geonor']

        else:
            df_fct_3.loc[time,'precip_inst_geonor'] = row['precip_inst_geonor']

    df_fct_3['precip_tot_geonor'] = df_fct_3['precip_inst_geonor'].cumsum()

    # plt.plot(df.index, df['precip_tot_geonor'], label='cas_3')

    return df_fct_3

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
    CE = a * np.e **(-b*wind) + c

    return CE

def calcul_wind_10m(df,z_1,z_2):
    z_0 = 0.01
    d = 0.4
    u_z1 = df['vitesse_vent_moy_2m']
    factor = np.log((z_2-d)/z_0)  / np.log((z_1-d)/z_0)
    u_z2 = u_z1 * factor
    return u_z2


def calcul_incertitue(df):


    pd.set_option('use_inf_as_na', True)
    df_30min = df.resample('0.5H', label='right').agg(resample_1h_op)
    df_1h = df.resample('1H', label='right').agg(resample_1h_op)

    # 30 min
    df_thresh_calc_30min = df_30min.copy()
    df_thresh_0p11_30min = df_30min.copy()
    df_thresh_pluvio_30min = df_30min.copy()

    # non_zero_geo_30min = df_thresh_pluvio_30min['precip_inst_geonor'][~(df_thresh_pluvio_30min['precip_inst_geonor'] == 0)]
    # non_zero_pluvio_30min = df_thresh_pluvio_30min['precip_inst_pluvio'][~(df_thresh_pluvio_30min['precip_inst_pluvio'] == 0)]


    df_thresh_pluvio_30min.loc[df_thresh_pluvio_30min['precip_inst_pluvio'] < 0, ['precip_inst_pluvio']] = 0

    wind_10m = calcul_wind_10m(df_thresh_pluvio_30min,2,10)
    df_thresh_pluvio_30min['vitesse_vent_moy_10m'] = wind_10m


    df_thresh_pluvio_30min_windtemp = df_thresh_pluvio_30min[(df_thresh_pluvio_30min['temp_moy']<-2) & (df_thresh_pluvio_30min['vitesse_vent_moy_10m']>5) &  (df_thresh_pluvio_30min['vitesse_vent_moy_10m']<9)]
    non_zero_geo_30min = df_thresh_pluvio_30min_windtemp['precip_inst_geonor'][
        ~(df_thresh_pluvio_30min_windtemp['precip_inst_geonor'] == 0)]
    non_zero_pluvio_30min = df_thresh_pluvio_30min_windtemp['precip_inst_pluvio'][
        ~(df_thresh_pluvio_30min_windtemp['precip_inst_pluvio'] == 0)]

    # threshold_calc_30min = np.median(non_zero_geo_30min.sort_values())/np.median(non_zero_pluvio_30min.sort_values()) * 0.25
    threshold_calc_30min_2 = np.median((non_zero_geo_30min.sort_values() / non_zero_pluvio_30min.sort_values()).dropna()) * 0.25

    print(f'30 min threshold: {threshold_calc_30min_2}')

    df_thresh_calc_30min.loc[df_thresh_calc_30min['precip_inst_geonor'] < threshold_calc_30min_2, ['precip_inst_geonor']] = 0
    df_thresh_calc_30min.loc[df_thresh_calc_30min['precip_inst_pluvio'] < 0.25, ['precip_inst_pluvio']] = 0
    df_thresh_0p11_30min.loc[df_thresh_0p11_30min['precip_inst_geonor'] < 0.11, ['precip_inst_geonor']] = 0

    plt.plot(df_1h.index, df_1h['precip_inst_pluvio'].cumsum(), label='DFAR no Threshold', path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],zorder = 999999)
    plt.plot(df_1h.index, df_1h['precip_inst_geonor'].cumsum(), label='SA no Threshold\n', path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],zorder = 999999)
    plt.plot(df_thresh_pluvio_30min.index, df_thresh_pluvio_30min['precip_inst_pluvio'].cumsum(),
             label='DFAR Threshold 0.25mm/30min (Kochendorfer et al. 2017 (1))', path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],zorder = 999999)
    # plt.plot(df_thresh_calc_30min.index, df_thresh_calc_30min['precip_inst_geonor'].cumsum(),
    #          label=f'SA Threshold calcul {threshold_calc_30min_2:.2f}mm/30min (Kochendorfer et al. 2017 (1))', path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],zorder = 999999)

    # plt.plot(df_thresh_0p11_30min.index, df_thresh_0p11_30min['precip_inst_geonor'].cumsum(),
    #          label='SA Threshold 0.11mm/30min (Kochendorfer et al. 2017 (2))\n',alpha=0.7)

    # 1h

    df_thresh_calc_1h = df_1h.copy()
    df_thresh_0p22_1h = df_1h.copy()
    df_thresh_pluvio_1h = df_1h.copy()
    df_thresh_pluvio_1h_1mm = df_1h.copy()

    df_thresh_pluvio_1h.loc[df_thresh_pluvio_1h['precip_inst_pluvio'] < 0.5, ['precip_inst_pluvio']] = 0
    df_thresh_pluvio_1h_1mm.loc[df_thresh_pluvio_1h_1mm['precip_inst_pluvio'] < 1, ['precip_inst_pluvio']] = 0

    non_zero_geo_1h = df_thresh_pluvio_1h['precip_inst_geonor'][~(df_thresh_pluvio_1h['precip_inst_geonor'] == 0)]
    non_zero_pluvio_1h = df_thresh_pluvio_1h['precip_inst_pluvio'][~(df_thresh_pluvio_1h['precip_inst_pluvio'] == 0)]
    threshold_calc_1h = np.median(non_zero_geo_1h.sort_values())/np.median(non_zero_pluvio_1h.sort_values())*0.25
    # threshold_calc_1h = np.median((non_zero_geo_1h.sort_values() / non_zero_pluvio_1h.sort_values()).dropna()) * 0.25
    # print(f'1 hour threshold: {threshold_calc_1h}')

    # df_thresh_calc_1h.loc[df_thresh_calc_1h['precip_inst_geonor'] < threshold_calc_1h, ['precip_inst_geonor']] = 0

    df_thresh_0p22_1h.loc[df_thresh_0p22_1h['precip_inst_geonor'] < 0.22, ['precip_inst_geonor']] = 0





    # plt.plot(df_thresh_pluvio_1h.index, df_thresh_pluvio_1h['precip_inst_pluvio'].cumsum(), label='DFAR Threshold 0.5mm/1h',alpha=0.7)
    # plt.plot(df_thresh_pluvio_1h_1mm.index, df_thresh_pluvio_1h_1mm['precip_inst_pluvio'].cumsum(),
    #          label='DFAR Threshold 1mm/1h (Pierre et al. 2019)\nPour event',alpha=0.7)
    # plt.plot(df_thresh_calc_1h.index,df_thresh_calc_1h['precip_inst_geonor'].cumsum(),label=f'SA Threshold calcul {threshold_calc_1h:.2f}/1h')
    plt.plot(df_thresh_0p22_1h.index, df_thresh_0p22_1h['precip_inst_geonor'].cumsum(), label='SA Threshold 0.22mm/1h (Pierre et al. 2019)\nPour event',alpha=0.7)
    plt.legend()
    plt.show()

    df_thresh_calc_30min = df_thresh_calc_30min.resample('1H', label='right').agg(resample_1h_op)
    df_thresh_calc_30min['precip_tot_geonor'] = df_thresh_calc_30min['precip_inst_geonor'].cumsum()
    df_thresh_calc_30min['precip_tot_pluvio'] = df_thresh_calc_30min['precip_inst_pluvio'].cumsum()
    df_thresh_0p22_1h['precip_tot_geonor'] = df_thresh_0p22_1h['precip_inst_geonor'].cumsum()
    # df_thresh_calc_30min.loc[df_thresh_calc_30min['precip_inst_pluvio'] < 0.25, ['precip_inst_pluvio']] = 0
    return df_thresh_calc_30min,df_thresh_0p22_1h


if __name__ == '__main__':


    for filename in all_files:
        print('Traitement de la sous-captation test ' + filename)
        df_momo_1 = pd.read_csv(filename, parse_dates=['date'])
        df_momo_1 = df_momo_1.set_index('date')
        df_momo_1.sort_index(inplace=True)

        # df_momo,df_threshold_0p22 = calcul_incertitue(df_momo_1)
        # threshold geonor obs
        df_momo_1.loc[df_momo_1['precip_inst_geonor'] < 0.2, ['precip_inst_geonor']] = 0

        # threshold geonor
        df_momo_1.loc[df_momo_1['precip_inst_geonor'] > 75, ['precip_inst_geonor']] = np.nan

        df_momo = df_momo_1.resample('1H', label='right').agg(resample_1h_op)

        df_momo['precip_tot_geonor'] = df_momo['precip_inst_geonor'].cumsum()
        df_momo['precip_tot_pluvio'] = df_momo['precip_inst_pluvio'].cumsum()

        tot_geo = df_momo['precip_tot_geonor'][-1]
        tot_pluvio = df_momo['precip_tot_pluvio'][-1]
        pluvio_arr= df_momo['precip_inst_pluvio'].values

        fig = plt.figure(facecolor='white', dpi=200, figsize=(7, 5))
        # figsize=figsize
        # Set projection defined by the cartopy object
        ax = fig.add_subplot()
        ax.plot(df_momo.index, df_momo['precip_tot_pluvio'], label='DFAR')
        ax.plot(df_momo.index, df_momo['precip_tot_geonor'], label='GEONOR')




        "______________"
        print('Case 1')
        #
        df_cas_1 =df_momo
        df_cas_1 = cas_1(df_cas_1)
        cas_1_arr = df_cas_1['precip_inst_geonor'].values

        RMSE_cas_1 = metrics.mean_squared_error(pluvio_arr, cas_1_arr, squared=False)
        bias_cas_1 = smt.eval_measures.bias(pluvio_arr, cas_1_arr)
        r_cas_1 = statis.pearsonr(pluvio_arr, cas_1_arr)[0]
        spearmanr_cas_1 = statis.spearmanr(pluvio_arr, cas_1_arr)[0]

        print(f"Added precip: {(df_cas_1['precip_tot_geonor'][-1] - tot_geo)*100 / tot_geo:.2f} %")
        print(f"Added precip: {(df_cas_1['precip_tot_geonor'][-1] - tot_geo):.2f} mm")

        if tot_pluvio - df_cas_1['precip_tot_geonor'][-1] < 0:
            print(f"Too much precip by : {-(tot_pluvio - df_cas_1['precip_tot_geonor'][-1]):.2f} mm")
            print(f"Too much precip by : {-(tot_pluvio - df_cas_1['precip_tot_geonor'][-1])*100/tot_pluvio:.2f} %\n")
        else:
            print(f"Still missing of : {(tot_pluvio - df_cas_1['precip_tot_geonor'][-1]):.2f} mm\n")
        print(f"Statistique:r_per={r_cas_1:.3f}, "
              f"r_spear={spearmanr_cas_1:.3f}, "
              f"bias={bias_cas_1:.3f} mm, "
              f"RMSE={RMSE_cas_1:.3f} mm\n")

        ax.plot(df_cas_1.index, df_cas_1['precip_tot_geonor'], label='cas 1 Eq. 3')

        # ax.plot(df_momo_1.index, df_momo_1['precip_tot_pluvio'], label = 'DFAR no threshold')
        "______________"
        print('Case 2')
        df_cas_2 = df_momo
        df_cas_2 = cas_2(df_cas_2)
        cas_2_arr = df_cas_2['precip_inst_geonor'].values

        RMSE_cas_2 = metrics.mean_squared_error(pluvio_arr, cas_2_arr, squared=False)
        bias_cas_2 = smt.eval_measures.bias(pluvio_arr, cas_2_arr)
        r_cas_2 = statis.pearsonr(pluvio_arr, cas_2_arr)[0]
        spearmanr_cas_2 = statis.spearmanr(pluvio_arr, cas_2_arr)[0]

        print(f"Added precip: {(df_cas_2['precip_tot_geonor'][-1] - tot_geo)*100  / tot_geo:.2f} %")
        print(f"Added precip: {(df_cas_2['precip_tot_geonor'][-1] - tot_geo):.2f} mm")
        if tot_pluvio - df_cas_2['precip_tot_geonor'][-1] < 0:
            print(f"Too much precip : {-(tot_pluvio - df_cas_2['precip_tot_geonor'][-1]):.2f} mm")
            print(f"Too much precip by : {-(tot_pluvio - df_cas_2['precip_tot_geonor'][-1]) * 100 / tot_pluvio:.2f} %\n")
        else:
            print(f"Still missing of : {(tot_pluvio - df_cas_2['precip_tot_geonor'][-1]):.2f} mm\n")
        print(f"Statistique:r_per={r_cas_2:.3f}, "
              f"r_spear={spearmanr_cas_2:.3f}, "
              f"bias={bias_cas_2:.3f} mm, "
              f"RMSE={RMSE_cas_2:.3f} mm\n")

        ax.plot(df_cas_2.index, df_cas_2['precip_tot_geonor'], label='cas 2 Eq. 4')


        "______________"
        print('Case 3')

        # df_cas_3 = df_momo
        # df_cas_3 = cas_3(df_cas_3)
        # cas_3_arr = df_cas_3['precip_inst_geonor'].values
        #
        # RMSE_cas_3 = metrics.mean_squared_error(pluvio_arr, cas_3_arr, squared=False)
        # bias_cas_3 = smt.eval_measures.bias(pluvio_arr, cas_3_arr)
        # r_cas_3 = statis.pearsonr(pluvio_arr, cas_3_arr)[0]
        # spearmanr_cas_3 = statis.spearmanr(pluvio_arr, cas_3_arr)[0]
        #
        # print(f"Added precip: {(df_cas_3['precip_tot_geonor'][-1]-tot_geo)*100 /tot_geo:.2f} %")
        # print(f"Added precip: {(df_cas_3['precip_tot_geonor'][-1]- tot_geo)} mm")
        # if tot_pluvio - df_cas_3['precip_tot_geonor'][-1] <0:
        #     print(f"Too much precip : {-(tot_pluvio - df_cas_3['precip_tot_geonor'][-1])} mm")
        #     print(f"Too much precip by : {-(tot_pluvio - df_cas_3['precip_tot_geonor'][-1]) * 100 / tot_pluvio} mm\n")
        #
        # else:
        #     print(f"Still missing of : {(tot_pluvio - df_cas_3['precip_tot_geonor'][-1])} mm\n")
        #
        # print(f"Statistique:r_per={r_cas_3:.3f}, "
        #       f"r_spear={spearmanr_cas_3:.3f}, "
        #       f"bias={bias_cas_3:.3f}, "
        #       f"RMSE={RMSE_cas_3:.3f}\n")
        # ax.plot(df_cas_3.index, df_cas_3['precip_tot_geonor'], label='cas 3')



        plt.legend()
        # locator_minor = mdates.HourLocator(interval=24 * 30)
        # locator_major = mdates.HourLocator(interval=24 * 365)
        ax.set_xlabel('Time [UTC]', fontsize=10)
        ax.set_ylabel('Total accumulation [mm]', fontsize=10)


        # ax.xaxis.set_major_locator(locator_major)
        # ax.xaxis.set_minor_locator(locator_minor)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m\n%Y"))
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.tick_params(axis='x', which='minor', labelsize=7)

        fig.savefig(f'/Users/olivier1/Documents/GitHub/data_format-master/figures/analyse_fct_transfert.png', dpi=200,
                    format='png',
                    bbox_inches='tight')
        plt.show()

