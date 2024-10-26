# -*- coding: utf-8 -*-
"""

Created on Tue Jan 11 09:39:41 2022

Master script du formatage des données de disdromètres. Inclut toutes les
transformations pour passer des données brutes à des données prêtes à utiliser
en modélisation.

@author: alexi
"""

#%% Répertoire du projet et importation des modules utilisé#s
import os

project_path =  os.path.abspath(os.path.join(__file__ ,"../.."))




# module_path = os.path.join(project_path, r"data_format-master\Functions")
# sys.path.append(module_path)

import Functions.pre_format_database as pre_format_database
from Functions.hydromet_formating import hydromet_format_func
from Functions.formating_functions import format_func
from Functions.timestep_formating import pdt_15min_to_1h
from Functions.undercatch import undercatchement_hq


from Function_oli.hole_functions import dfholes,summary_file_1h
from Function_oli.event_finding_fct import event_finding_fct
from Function_oli.foret_momo import creation_dataset_momorency
from Function_oli.PK_dataset import creation_dataset_uqam
from Function_oli.stat_removing import stat_analysis_r


#%% Répertoire des bases de données initiales et de sauvegarde
HQ_datapath = os.path.join(project_path, r"data_format-master/Data.nosync/station_gmon/Raw/gmon-stations-raw")
# HQ_preformat_savepath = os.path.join(project_path,'data_format-master' ,'Data','station_meteo','stations','hydromet-station-raw')

# data raw mais faut que ça soit traiter une premier fois
HQ_hydromet_path = os.path.join(project_path, r"data_format-master/Data.nosync/station_meteo/Compiled_DB")

HQ_compiled_savepath = os.path.join(project_path, r"data_format-master/Data.nosync/station_gmon/Full_datasets")


# SN_datapath = os.path.join(project_path, r"Data/SN/Raw")
# SN_compiled_savepath = os.path.join(project_path, r"Data/SN/Compiled_DB")

#%% Nettoyage des données hydrométéo
# format_hydromet = True
format_hydromet = False
if format_hydromet:
    hydromet_format_func(os.path.join(project_path,'data_format-master','Data.nosync','station_meteo','stations','hydromet-station-raw'),
                       HQ_hydromet_path)

#%% Uniformisation du jeu de données
# reload = True
reload = False
if reload:
    # pre_format_database.HQ(HQ_datapath, HQ_preformat_savepath, HQ_hydromet_path)
    pre_format_database.HQ(HQ_datapath, HQ_compiled_savepath, HQ_hydromet_path)

    # pre_cleanup(HQ_preformat_savepath, HQ_compiled_savepath)

#%% Gap-filling des données hydrométéo
# TODO adapter selon les besoins cernés
# gapfill = True
# gapfill = False

# data_path = os.path.join(df_savepath, 'FULL_DATASET.csv')
# hydromet_path = os.path.join(project_path, 'Data', 'HQ',
#                              'Hydrometeo', 'hydro.2022.04.12.csv')
# save_path = os.path.join(project_path, r"Data\HQ")

# if gapfill:
#     hydromet_gapfill(data_path, hydromet_path, save_path)

#%% Clean-up des différents types de mesures
# Potentiellement il y a une interpolation qu'il faudra que je change ou enleve.


cleanup = True
# cleanup = False
if cleanup:
    format_func(HQ_compiled_savepath)




#%% Création des bases de données aux différents pas de temps
# changement_pdt = True
changement_pdt = False
if changement_pdt:
    pdt_15min_to_1h(HQ_compiled_savepath)


# undercatchement
# undercatch = True
undercatch = False
if undercatch:
    undercatchement_hq(HQ_compiled_savepath)

# analyse_qty_lufft = True
analyse_qty_lufft = False
if analyse_qty_lufft:
    stat_analysis_r(HQ_compiled_savepath)


hole = False
if hole:
    dfholes(HQ_compiled_savepath)
    summary_file_1h(HQ_compiled_savepath,'/Users/olivier1/Documents/GitHub/data_format-master/Data.nosync/Stations_HQ.xlsx')


# find_prcp_event = True
find_prcp_event = False
if find_prcp_event:
    event_finding_fct(HQ_compiled_savepath)



# montmonrency_forest = False
montmonrency_forest = True
path_data_momo = os.path.join(project_path, r"data_format-master/Data.nosync/site_neige/new_data")
savingpath_data_momo = os.path.join(project_path, r"data_format-master/Data.nosync/site_neige/Full_dataset")
if montmonrency_forest:
    creation_dataset_momorency(path_data_momo,savingpath_data_momo)



# uqam_pk = False
uqam_pk = True
path_data_momo = os.path.join(project_path, r"data_format-master/Data.nosync/site_uqam/new_data")
savingpath_data_momo = os.path.join(project_path, r"data_format-master/Data.nosync/site_uqam/Full_dataset")

if uqam_pk:
    creation_dataset_uqam(path_data_momo,savingpath_data_momo)



#%% Création d'une database unique
# path_list = [HQ_preformat_savepath]
# df_savepath = os.path.join(project_path, r"Data\HQ")
# df = df_load(path_list, df_savepath)

#%% Visualisation des données
# graph_savepath = os.path.join(project_path, r'Graphs\data_visualisation')

# graph = True
# graph = False
# if graph:
#     db_visual(df, graph_savepath)
