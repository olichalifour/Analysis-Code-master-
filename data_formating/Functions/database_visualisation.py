# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:47:02 2022

Fonctions de visualisation graphique des données d'entrées et de validation de
la base de données de disdromètres.

@author: alexi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools

#%% Main function
def db_visual(df, savepath):    
    
    functions = [precip_graph,
                  snow_graph, 
                  density_profile, 
                  temp_scatter, 
                  temp_graph,
                  humidite_rel,
                  frequency_stack]
    for func in functions:
        func(df, savepath)

    for temp_var in ['temp_moy', 'temp_max', 'temp_min']:
        for all_types in [True, False]:
            frequency_curve(df, temp_var, all_types, savepath)            
    
    return

#%% frequency curve
def frequency_curve(df, temp_var, all_types, savepath):    
    
    bins = np.linspace(-10, 10, num=41)
    types = df.groupby(['type_precip', pd.cut(df[temp_var], bins)])
    types_df = types.size().unstack()
    types_df.drop(index = types_df.index[types_df.index == 0], inplace=True)
    
    plt.figure()
    if all_types:
       freq_solid = (types_df[types_df.index.isin([70])]
                       .sum(axis=0)
                       .div(types_df
                            .sum(axis=0))) 
   
       freq_liquid = (types_df[types_df.index.isin([60])]
                      .sum(axis=0)
                      .div(types_df
                           .sum(axis=0)))
       
       freq_melt = (types_df[types_df.index.isin([69])]
                    .sum(axis=0)
                    .div(types_df
                         .sum(axis=0))) 

       freq_freezrain = (types_df[types_df.index.isin([67])]
                         .sum(axis=0)
                         .div(types_df
                              .sum(axis=0)))
   
       plt.plot(bins[:-1], freq_solid.values, label='Neige')
       plt.plot(bins[:-1], freq_liquid.values, label='Pluie')
       plt.plot(bins[:-1], freq_melt.values, label='Pluie/bruine et neige')
       plt.plot(bins[:-1], freq_freezrain.values, label='Pluie verglaçante')
       
       plt.legend()
       
    else:
        freq_solid = (types_df[types_df.index.isin([67, 69, 70])]
                        .sum(axis=0)
                        .div(types_df
                             .sum(axis=0))) 
    
        freq_liquid = (types_df[types_df.index.isin([60])]
                .sum(axis=0)
                .div(types_df
                     .sum(axis=0))) 
    
        plt.plot(bins[:-1], freq_solid.values, label='Solide')
        plt.plot(bins[:-1], freq_liquid.values, label='Liquide')
        
        plt.legend()
    
    temp_label = {'temp_moy': "Température moyenne de l'air ($\degree$C)",
                  'temp_max': "Température maximum horaire de l'air ($\degree$C)",
                  'temp_min': "Température minimum horaire de l'air ($\degree$C)"}
    plt.xlabel(temp_label.get(temp_var))
    plt.ylabel('Fréquence (-)')

    plt.title('Fréquence du type de phase lors d\'une précipitation\n' 
             + 'n = ' + str(types_df.sum().sum()))
    
    graph_type = ['all_phases_' if all_types else '']
    plt.savefig(os.path.join(savepath, 'Frequency_curve',
                             ('snow_frequency_' + graph_type[0] + temp_var + '.png')),
                dpi=300, bbox_inches='tight')
    plt.close()
    return

#%% frequency stack
def frequency_stack(df, savepath):
    
    temp_vars = ['temp_moy', 'temp_max', 'temp_min']
    phase_vars = [[67, 69, 70], [60]]    
    
    plt.figure()
    
    for phase in phase_vars:
        for temp in temp_vars:
            bins = np.linspace(-10, 10, num=41)
            types = df.groupby(['type_precip', pd.cut(df[temp], bins)])
            types_df = types.size().unstack()
            types_df.drop(index = types_df.index[types_df.index == 0], inplace=True)
            
            freq = (types_df[types_df.index.isin(phase)]
                            .sum(axis=0)
                            .div(types_df
                                 .sum(axis=0)))
            plt.plot(bins[:-1], freq.values, label=temp)
            
        plt.legend()
        plt.xlabel("Température de l'air ($\degree$C)")
        plt.ylabel('Fréquence (-)')
        
        phase_dict = ['liquides' if 60 in phase else 'solides']
        plt.title('Fréquence de précipitations ' + phase_dict[0] + ' selon la température\n' 
                 + 'n = ' + str(types_df.sum().sum()))    
        plt.savefig(os.path.join(savepath, 'Frequency_curve',
                                 ('frequency_curve_' + phase_dict[0] + '_only.png')),
                    dpi=300, bbox_inches='tight')
        plt.close()

    return

#%% precip
def precip_graph(df, savepath):
    print('Génération des graphiques de précipitation...')
    
    year_list = (df
                 .index
                 .year
                 .unique()
                 .sort_values())
        
    for year_start, year_end in zip(year_list[:-1], year_list[1:]):
        start = (str(year_start) + '-10-01')
        end = (str(year_end) + '-06-01')
        
        subdf = df[start: end]
        
        x_moy = subdf['precip_tot_pluvio'].groupby('date').mean()
        x_max = x_moy + subdf['precip_tot_pluvio'].groupby('date').std()
        x_min = x_moy - subdf['precip_tot_pluvio'].groupby('date').std()
        
        plt.figure()
        plt.fill_between(x_max.index, x_max, x_min, color='C0', alpha=0.3)
        plt.plot(x_moy.index, x_moy, color='C0')
        
        plt.xticks(rotation=45)
        plt.title('Précipitation cumulative moyenne ± l\'écart-type')
        plt.ylabel('Précipitation cumulative (mm)')
        plt.savefig(os.path.join(savepath, 'Precipitation',
                                 ('precip_cumul_fulldataset_' +
                                  str(year_start) + '-' + str(year_end) + '.png')),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    
    # Précipitation cumulative par station
    for station, subdf in df.groupby('filename'):
        year_list = (subdf
                      .index
                      .year
                      .unique()
                      .sort_values())
        
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
           
            plt.figure()
            (subdf.loc[start:end, 'precip_tot_disdro']
             .sub(subdf.loc[start:end, 'precip_tot_disdro'].min())
             .plot(title=('Précipitation cumulative à la station ' 
                          + station), ylabel='Précipiation cumulative (mm)'))
            
            # if subdf['precip_tot_pluvio'].isna().sum() != subdf['precip_tot_pluvio'].size:
            (subdf.loc[start:end, 'precip_tot_pluvio']
             .sub(subdf.loc[start:end, 'precip_tot_pluvio'].min())
             .plot())
            plt.legend(['Disdromètre', 'Pluviomètre'])
            # else:
                # plt.legend(['Disdromètre'])
                
            plt.savefig(os.path.join(savepath, 'Precipitation', 'Stations', 
                                     ('precip_cumul_' + station
                                      + '_' + str(year_start) + '-' + str(year_end) + '.png')),
                        dpi=300, bbox_inches='tight')
            plt.close()
    print('Graphiques de précipitation générés.\n')
    return

#%% humidité relative
def humidite_rel(df, savepath):
    print("Génération des graphiques d'humidité...")
    
    year_list = (df
                 .index
                 .year
                 .unique()
                 .sort_values())
    
    for year_start, year_end in zip(year_list[:-1], year_list[1:]):
        start = (str(year_start) + '-10-01')
        end = (str(year_end) + '-06-01')
        
        subdf = df[start: end]
        
        x_moy = subdf['humidite_air'].groupby('date').mean()
        
        plt.figure()
        plt.plot(x_moy.index, x_moy, color='C0')
        
        plt.xticks(rotation=45)
        plt.title('Humidité relative pour toutes les stations')
        plt.ylabel('Humidité relative (%)')
        plt.savefig(os.path.join(savepath, 'Humidité',
                                 ('humidite_fulldataset_' +
                                  str(year_start) + '-' + str(year_end) + '.png')),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    
    for station, subdf in df.groupby('filename'):
        year_list = (subdf
                      .index
                      .year
                      .unique()
                      .sort_values())
        
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
           
            x_moy = subdf.loc[start: end, 'humidite_air']
            
            plt.figure()
            plt.plot(x_moy.index, x_moy, color='C0')
            
            plt.xticks(rotation=45)
            plt.title(station + ': humidité relative')
            plt.ylabel('Humidité relative (%)')           
                
            plt.savefig(os.path.join(savepath, 'Humidité', 'Stations', 
                                     ('humidite_' + station
                                      + '_' + str(year_start) + '-' + str(year_end) + '.png')),
                        dpi=300, bbox_inches='tight')
            plt.close()
    print('Graphiques de humidité relative générés.\n')
    return

#%% snow
def snow_graph(df, savepath):
    print('Génération des graphiques de neige...')
    
    snow_var = ['EEN_K', 'neige_sol']
    labelDict = {'EEN_K': 'EEN (mm)',
                 'neige_sol': 'Hauteur de neige (cm)'}
    titleDict = {'EEN_K': 'EEN',
                 'neige_sol': 'Hauteur de neige'}
    filenameDict = {'EEN_K': 'EEN',
                 'neige_sol': 'hauteur_neige'}
    
    for var in snow_var:
        year_list = (df
                     .index
                     .year
                     .unique()
                     .sort_values())
            
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
            
            subdf = df[start: end]
            
            x_moy = subdf[var].groupby('date').mean()
            x_max = x_moy + subdf[var].groupby('date').std()
            x_min = x_moy - subdf[var].groupby('date').std()
            
            plt.figure()
            plt.fill_between(x_max.index, x_max, x_min, color='C0', alpha=0.3)
            plt.plot(x_moy.index, x_moy, color='C0')
            
            plt.xticks(rotation=45)
            plt.title((titleDict.get(var) + ' moyenne ± l\'écart-type'))
            plt.ylabel(labelDict.get(var))
            plt.savefig(os.path.join(savepath, 'Snow',
                                     (filenameDict.get(var) + '_fulldataset_' +
                                      str(year_start) + '-' + str(year_end) + '.png')),
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        
    for station, subdf in df.groupby('filename'):
        year_list = (subdf
                      .index
                      .year
                      .unique()
                      .sort_values())
        
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
           
            fig, ax1 = plt.subplots()
            subdf.loc[start:end, 'EEN_K'].plot(ax=ax1, label='SWE')
            ax2 = ax1.twinx()
            ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler
            subdf.loc[start:end, 'neige_sol'].plot(ax=ax2, label='Hauteur de neige')
            
            ax1.set_ylabel('EEN (mm)')
            ax1.tick_params(axis='y', labelcolor='C0')
            
            ax2.set_ylabel('Hauteur de neige (cm)')
            ax2.tick_params(axis='y', labelcolor='C1')
            
            plt.title('EEN et hauteur de neige à la station ' + station)            
            plt.savefig(os.path.join(savepath, 'Snow', 'Stations', 
                                         ('SWE_hauteur_neige_' + station
                                          + '_' + str(year_start) + '-' + str(year_end) + '.png')),
                            dpi=300, bbox_inches='tight')
            plt.close()
    print('Graphiques de neige générés.\n')
    return

#%% density profile
def density_profile(df, savepath):
    print('Génération des profils de densité...')

    year_list = (df
                 .index
                 .year
                 .unique()
                 .sort_values())
        
    for year_start, year_end in zip(year_list[:-1], year_list[1:]):
        start = (str(year_start) + '-10-01')
        end = (str(year_end) + '-06-01')
        
        subdf = df[start: end]
        
        dens = (subdf['neige_sol']
                .div(10)
                .div(subdf['EEN_K']))
        x_moy = dens.groupby('date').mean()
        x_max = x_moy + dens.groupby('date').std()
        x_min = x_moy - dens.groupby('date').std()
        
        plt.figure()
        plt.fill_between(x_max.index, x_max, x_min, color='C0', alpha=0.3)
        plt.plot(x_moy.index, x_moy, color='C0')
        
        plt.xticks(rotation=45)
        plt.title(('Densité relative moyenne ± l\'écart-type'))
        plt.ylabel('Densité relative (-)')
        plt.savefig(os.path.join(savepath, 'Snow',
                                 ('density_profile_fulldataset_' +
                                  str(year_start) + '-' + str(year_end) + '.png')), 
                            dpi=300, bbox_inches='tight')
        plt.close()
    
    
    for station, subdf in df.groupby('filename'):
        year_list = (subdf
                      .index
                      .year
                      .unique()
                      .sort_values())
        
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
           
            plt.figure() 
            dens = (subdf.loc[start: end, 'neige_sol']
                    .div(subdf.loc[start: end, 'EEN_K']))
            
            plt.plot(dens.index, dens, color='C0')
            
            plt.xticks(rotation=45)
            plt.title((station + ': densité relative'))
            plt.ylabel('Densité relative (-)')
            
            plt.savefig(os.path.join(savepath, 'Snow', 'Stations', 
                                         ('density_profile' + station
                                          + '_' + str(year_start) + '-' + str(year_end) + '.png')),
                            dpi=300, bbox_inches='tight')
            plt.close()
            
    print('Profils de densité générés.\n')
    return

#%% temperature
def temp_graph(df, savepath):
    print('Génération des graphiques de température...')
    
    year_list = (df
                 .index
                 .year
                 .unique()
                 .sort_values())
    
    for year_start, year_end in zip(year_list[:-1], year_list[1:]):
        start = (str(year_start) + '-10-01')
        end = (str(year_end) + '-06-01')
        
        subdf = df[start: end]
        
        x_moy = subdf['temp_moy'].groupby('date').mean()
        
        plt.figure()
        plt.plot(x_moy.index, x_moy, color='C0')
        
        plt.xticks(rotation=45)
        plt.title('Température moyenne pour toutes les stations')
        plt.ylabel('Température ($\degree$C)')
        plt.savefig(os.path.join(savepath, 'Temperature',
                                 ('temperature_fulldataset_' +
                                  str(year_start) + '-' + str(year_end) + '.png')),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    
    for station, subdf in df.groupby('filename'):
        year_list = (subdf
                      .index
                      .year
                      .unique()
                      .sort_values())
        
        for year_start, year_end in zip(year_list[:-1], year_list[1:]):
            start = (str(year_start) + '-10-01')
            end = (str(year_end) + '-06-01')
           
            for var in ['temp_moy', 'temp_max', 'temp_min']:
                x_moy = subdf.loc[start: end, var]
                
                plt.figure()
                plt.plot(x_moy.index, x_moy, color='C0')
                
                plt.xticks(rotation=45)
                plt.title(station + ': ' + var)
                plt.ylabel('Température ($\degree$C)')           
                    
                plt.savefig(os.path.join(savepath, 'Temperature', 'Stations', 
                                         (var + '_' + station
                                          + '_' + str(year_start) + '-' + str(year_end)
                                          + '.png')),
                            dpi=300, bbox_inches='tight')
                plt.close()
    print('Graphiques de température générés.\n')
    return

#%% temperature scatter
# TODO modifier pour donner plus d'infos
def temp_scatter(df, savepath):
    print('Génération des nuages de points des variables d\'entrée...')
    
    solidPrecip = df.loc[df['type_precip'].isin([67, 69, 70]),
                         ['filename', 'temp_moy', 'temp_max', 'temp_min']]
    liquidPrecip = df.loc[df['type_precip'].isin([60]), 
                         ['filename', 'temp_moy', 'temp_max', 'temp_min']]    
    
    
    labelDict = {'temp_moy': 'Temp_moy ($\degree$C)', 
                 'temp_max': 'Temp_max ($\degree$C)', 
                 'temp_min': 'Temp_min ($\degree$C)'}    
    pairs = list(itertools.combinations(labelDict.keys(), 2))
    
    # Scatter pour toutes les stations
    fig, subfigs = plt.subplots(1, len(pairs))
    for (x, y), subfig in zip(pairs, subfigs.reshape(-1)):
        subfig.scatter(solidPrecip[x], solidPrecip[y], s=5, label='Solide')
        subfig.scatter(liquidPrecip[x], liquidPrecip[y], s=5, label='Liquide')
        
        subfig.set(xlabel=labelDict.get(x), ylabel=labelDict.get(y))

    fig.subplots_adjust(bottom=0.3, wspace=0.75)
    fig.suptitle('Température selon la phase de la précipitation')
    subfigs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False,
                      ncol=3)
    plt.savefig(os.path.join(savepath, 'Temperature', 'Scatter_temp_input_fulldataset.png'), dpi=300)
    plt.close()   
    
    # Scatter par station
    for station, subdf in df.groupby('filename'):
        solidPrecip = subdf.loc[subdf['type_precip'].isin([67, 69, 70]),
                                ['filename', 'temp_moy', 'temp_max', 'temp_min']]
        liquidPrecip = subdf.loc[subdf['type_precip'].isin([60]),
                                ['filename', 'temp_moy', 'temp_max', 'temp_min']]
        
        fig, subfigs = plt.subplots(1, len(pairs))
        for (x, y), subfig in zip(pairs, subfigs.reshape(-1)):
            subfig.scatter(solidPrecip[x], solidPrecip[y], s=5, label='Solide')
            subfig.scatter(liquidPrecip[x], liquidPrecip[y], s=5, label='Liquide')
            
            subfig.set(xlabel=labelDict.get(x), ylabel=labelDict.get(y))
    
        fig.subplots_adjust(bottom=0.3, wspace=0.75)
        fig.suptitle('Station ' + station + 
                     ':\n Permutations de variables d\'entré selon la phase de la précipitation')
        subfigs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False,
                          ncol=2)
        plt.savefig(os.path.join(savepath, 'Temperature', 'Stations',
                                 'Scatter_temp_input_' + station + '.png'), dpi=300)
        plt.close()
    
    print('Nuages de points des variables d\'entrée générés.\n')
    return

#%% stations_pluvio
def stations_pluvio(df, savepath):
    # df = df.loc[df['pluviometer_type'].notna()]
    
    var_list = ['temp_max', 'temp_min', 'temp_moy', 
                'humidite_air', 'precip_inst_pluvio']
    pairs = list(itertools.combinations(var_list, 2))
    
    solidPrecip = df.loc[df['type_precip'].isin([67, 69, 70])]
    liquidPrecip = df.loc[df['type_precip'].isin([60])]
    
    labelDict = {'temp_moy': 'Température moyenne ($\degree$C)', 
                 'temp_max': 'Température maximum ($\degree$C)', 
                 'temp_min': 'Tempétaure minimum ($\degree$C)',
                 'humidite_air': 'Humidité relative (-)',
                 'precip_inst_pluvio': 'Précipitation (mm)'}
    
    
    for (x, y) in pairs:
        fig, ax = plt.subplots()
        ax.scatter(solidPrecip[x], solidPrecip[y], s=5, label='Solide')
        ax.scatter(liquidPrecip[x], liquidPrecip[y], s=5, label='Liquide')
        
        ax.set(xlabel=labelDict.get(x), ylabel=labelDict.get(y))
        
        plt.legend()
        fig.subplots_adjust(bottom=0.3, wspace=0.75)    
        fig.tight_layout()

        plt.savefig(os.path.join(savepath, 'Stations_pluvio',
                                 ('Scatter_input_vars_%s_%s.png' %(x, y))), dpi=300)
        plt.close()
        
    # Scatter par station
    df = df.loc[df['pluviometer_type'].notna()]
    for station, subdf in df.groupby('filename'):
        solidPrecip = subdf.loc[subdf['type_precip'].isin([67, 69, 70])]
        liquidPrecip = subdf.loc[subdf['type_precip'].isin([60])]
        
        for (x, y)in pairs:
            fig, ax = plt.subplots()
            ax.scatter(solidPrecip[x], solidPrecip[y], s=5, label='Solide')
            ax.scatter(liquidPrecip[x], liquidPrecip[y], s=5, label='Liquide')
            
            ax.set(xlabel=labelDict.get(x), ylabel=labelDict.get(y))
            
            fig.suptitle('Station ' + station)
            plt.legend()
            plt.savefig(os.path.join(
                savepath, 'Stations_pluvio', 'Stations','Scatter_input_'
                + station + ('_%s_%s.png' %(x, y))), dpi=300)
            plt.close()
    
    return

#%% input variables
def input_vars(df, savepath):
    phase_list = [70, 60, 69, 67]
    phase_str = ['Neige', 'Pluie', 'Pluie/bruine et neige', 'Pluie verglaçante']
    phaseDict = {}
    for key, label in zip(phase_list, phase_str):
        phaseDict[key] = label
            
    # Scatter par phase
    fig, ax = plt.subplots(2, 2)
    for phase, color, subfig in zip(phase_list, ['C0', 'C1', 'C2', 'C3'],
                                    ax.ravel()):        
        
        x = df.loc[df['type_precip'] == phase, 'temp_moy']
        y = df.loc[df['type_precip'] == phase, 'humidite_air']
        subfig.scatter(x, y, s=2.5, c=color, alpha=0.8)
        
        subfig.set_title(phaseDict.get(phase))
        subfig.set_xlabel("Température moyenne de l'air ($\degree$C)")
        subfig.set_ylabel("Humidité relative (%)")
    
    fig.tight_layout()
    plt.savefig(os.path.join(savepath, 'Input_variables', 'input_vars_scatter.png'), dpi=300)
    plt.close()    
    
    # Scatter par stations
    for station, subdf in df.groupby(['filename', 'pluviometer_type']):
        fig, ax = plt.subplots(2, 2)
        for phase, color, subfig in zip(phase_list, ['C0', 'C1', 'C2', 'C3'],
                                        ax.ravel()):        
            
            x = subdf.loc[subdf['type_precip'] == phase, 'temp_moy']
            y = subdf.loc[subdf['type_precip'] == phase, 'humidite_air']
            subfig.scatter(x, y, s=2.5, c=color, alpha=0.8)
            
            subfig.set_title(phaseDict.get(phase))
            subfig.set_xlabel("Température moyenne de l'air ($\degree$C)")
            subfig.set_ylabel("Humidité relative (%)")
        
        fig.suptitle('Station %s' %station[0])
        fig.tight_layout()
        plt.savefig(os.path.join(
            savepath, 'Input_variables', 'Stations','input_vars_scatter_'
            + station[0] + '.png'), dpi=300)
        plt.close()
        
    return