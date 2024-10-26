#!/usr/bin/env python3
#
# Before opening Python you need to first load the Python version (in this case Python3) you want to use with:
#   module load python3/miniconda3
# and then load the module needed to read RPN files with Python with:
#   module load python3/python-rpn
#   module load python3/outils-divers
# and to get access to basic Python packages source:
#   source activate base_plus
#

# Importation of modules and library
import numpy as np  # Good module for matrix and matrix operation
import matplotlib.pyplot as plt  # Module to produce figure
import matplotlib.colors as colors
import os
import glob

# Used to convert png to other format
try:
    import rpnpy.librmn.all as rmn  # Module to read RPN files
    from rotated_lat_lon import RotatedLatLon  # Module to project field on native grid (created by Sasha Huziy)
except ImportError as err:
    print(f"RPNPY can only be use on the server. It can't be use on a personal computer."
          f"\nError throw :{err}")
import pandas as pd
import cartopy.crs as ccrs  # Import cartopy ccrs
import cartopy.feature as cfeature  # Import cartopy common features
import math

lat_qc = 46.820634
lon_qc = -71.232010
lat_momo = 47.322437368331876
lon_momo = -71.14730110000002

# station radiosondage
# maniwaki,sept-ile,
lat_sond = [46.3019, 50.2233]
lon_sond = [-76.0061, -66.2656]

# station climat sentinel
# name = ['Gault', 'Arboretum', 'PK-UQAM', 'Trois-Rivieres', 'Sorel']
# list_stat_lon = [-73.149006, -73.942156, -73.568741, -72.581354, -73.110328]
# list_stat_lat = [45.535021, 45.430065, 45.508594, 46.349835, 46.030244]
name = ['Gault', 'Arboretum', 'PK-UQAM', ]
list_stat_lon = [-73.149006, -73.942156, -73.568741]
list_stat_lat = [45.535021, 45.430065, 45.508594]
# stattin HQ

df_disdro = pd.read_csv('/upslope/chalifour/code_fig_bassin_domaine/Disdrometres_coordonnées.csv', header=0)
df_hq = pd.read_csv('/upslope/chalifour/code_fig_bassin_domaine/stat_retra.csv', delimiter=';', header=0)

# Example parameters which need to get adjusted !!!
# ================================================
filename = '/pampa/poitras/DATA/chalifour/NAM11.fst'  # Name of RPN file to read
varname = 'MSKC'  # Name of variable to read
title = 'Domaines'
# unit     = r"${\rm ^\circ C}$"     # r + symbol of the unit of the values ("m", "%", "${\rm ^\circ C}$", "m/s", "${\rm W/m^2}$", ...)
# val_min  =  -4
# val_max  =  18
# val_int  =   2
# Output image information
image_output_file = "/upslope/chalifour/projet_maitrise/fig/fig_domain.png"
image_output_dpi = 200
# ================================================


# Read one record
# ---------------
fid = rmn.fstopenall(filename, rmn.FST_RO)  # Open the file
rec = rmn.fstlir(fid, nomvar=varname)  # Read the full record of variable 'varname'
field = rec['d']  # Assign 'field' to the data of 'varname'

# Read 2-D latitudes & longitudes - if needed
mygrid = rmn.readGrid(fid, rec)  # Get the grid information for the (LAM) Grid -- Reads the tictac's
latlondict = rmn.gdll(mygrid)  # Create 2-D lat and lon fields from the grid information
lat_12km = latlondict['lat']  # Assign 'lat' to 2-D latitude field
lon_12km = latlondict['lon']  # Assign 'lon' to 2-D longitude field

# Creer le rectangle
segSlon_12km = lon_12km[:, 0];
segSlat_12km = lat_12km[:, 0];  # segment sud   du domaine
segWlon_12km = lon_12km[-1, :];
segWlat_12km = lat_12km[-1, :];  # segment ouest du domaine
segNlon_12km = np.flip(lon_12km[:, -1]);
segNlat_12km = np.flip(lat_12km[:, -1]);  # segment nord  du domaine
segElon_12km = np.flip(lon_12km[0, :]);
segElat_12km = np.flip(lat_12km[0, :]);  # segment est  du domaine

rect12kmlon = np.concatenate((segSlon_12km, segWlon_12km, segNlon_12km, segElon_12km))
rect12kmlat = np.concatenate((segSlat_12km, segWlat_12km, segNlat_12km, segElat_12km))

# Get grid rotation for projection of 2-D field for mapping - if needed
tics = rmn.fstlir(fid, nomvar='^^', ip1=rec['ig1'], ip2=rec['ig2'], ip3=rec['ig3'])  # Read corresponding tictac's

# Close RPN input file
rmn.fstcloseall(fid)  # Close the RPN file

# 2-D Mapping - if needed
# -----------------------
# Get positions of rotated equator from IG1-4 of the tictac's
(Grd_xlat1, Grd_xlon1, Grd_xlat2, Grd_xlon2) = rmn.cigaxg('E', tics['ig1'], tics['ig2'], tics['ig3'], tics['ig4'])

# Use Sasha's RotatedLatLon to get the rotation matrix
rll = RotatedLatLon(lon1=Grd_xlon1, lat1=Grd_xlat1, lon2=Grd_xlon2,
                    lat2=Grd_xlat2)  # the params come from gemclim_settings.nml
# Use Sasha's get_cartopy_projection_obj to get the cartopy object for the projection and domain defined by the coordinates
m = rll.get_cartopy_projection_obj()

I = len(rect12kmlon)
rect12kmx = np.zeros(I)
rect12kmy = np.zeros(I)
for i in range(I):
    rect12kmx[i], rect12kmy[i] = m.transform_point(rect12kmlon[i], rect12kmlat[i], ccrs.PlateCarree())

##################################################################################################################################
filename = '/pampa/poitras/DATA/chalifour/E2p5.fst'  # Name of RPN file to read
fid = rmn.fstopenall(filename, rmn.FST_RO)  # Open the file
rec = rmn.fstlir(fid, nomvar=varname)  # Read the full record of variable 'varname'
field = rec['d']  # Assign 'field' to the data of 'varname'
# Read 2-D latitudes & longitudes - if needed
mygrid = rmn.readGrid(fid, rec)  # Get the grid information for the (LAM) Grid -- Reads the tictac's
latlondict = rmn.gdll(mygrid)  # Create 2-D lat and lon fields from the grid information
latE2p5 = latlondict['lat']  # Assign 'lat' to 2-D latitude field
lonE2p5 = latlondict['lon']  # Assign 'lon' to 2-D longitude field

# Creer le rectangle
segSlon = lonE2p5[:, 0];
segSlat = latE2p5[:, 0];  # segment sud   du domaine
segWlon = lonE2p5[-1, :];
segWlat = latE2p5[-1, :];  # segment ouest du domaine
segNlon = np.flip(lonE2p5[:, -1]);
segNlat = np.flip(latE2p5[:, -1]);  # segment nord  du domaine
segElon = np.flip(lonE2p5[0, :]);
segElat = np.flip(latE2p5[0, :]);  # segment est  du domaine

rectE2p5lon = np.concatenate((segSlon, segWlon, segNlon, segElon))
rectE2p5lat = np.concatenate((segSlat, segWlat, segNlat, segElat))

I = len(rectE2p5lon)
rectE2p5x = np.zeros(I)
rectE2p5y = np.zeros(I)
for i in range(I):
    rectE2p5x[i], rectE2p5y[i] = m.transform_point(rectE2p5lon[i], rectE2p5lat[i], ccrs.PlateCarree())

#################################################################################################################################


# Figure settings - if needed
# ---------------------------
# figsize = (5, 4.4)      # Figure size
fig = plt.figure()
# figsize=figsize
# Set projection defined by the cartopy object
ax = plt.axes(projection=m)

# Plotting - if needed
# --------------------

# Set corners of the maps
xll, yll = m.transform_point(lon_12km[0, 0], lat_12km[0, 0], ccrs.PlateCarree())
xur, yur = m.transform_point(lon_12km[-1, -1], lat_12km[-1, -1], ccrs.PlateCarree())

# Set geographic features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.5)  # couche ocean
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none')  # couche land
# ax.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
ax.add_feature(cfeature.BORDERS.with_scale('50m'))  # couche frontieres
# ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières
coast = cfeature.NaturalEarthFeature(category='physical', scale='10m', facecolor='none',
                                     name='coastline')  # Couche côtières
ax.add_feature(coast, edgecolor='black')
ax.add_feature(cfeature.LAKES)
states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m',
                                                facecolor='none')  # Couche provinces
ax.add_feature(states_provinces, edgecolor='grey')

ax.set_extent([xll - 5, xur + 5, yll - 5, yur + 5], crs=m)

# To help the layout of the figure after saving
# fig.canvas.draw()
plt.tight_layout()  # To help with the layout of the figure after saving

# plot station


# Plot domains
# plt.plot(xm,ym,'ro')
plt.plot(rect12kmx, rect12kmy, 'k--', lw=2, zorder=99999999, label='CORDEX 0.11\u00b0', )
plt.plot(rectE2p5x, rectE2p5y, 'b--', lw=2, zorder=99999999, label='Extended East 0.0225\u00b0')
point_momo = ax.scatter(lon_momo, lat_momo, facecolor='red', s=120, marker="*", label='Forêt Montmorency', zorder=9999,
                        edgecolors='k', transform=ccrs.PlateCarree())
# point_sondage = ax.scatter(lon_sond, lat_sond, facecolor='red', s=20, label='Radiosondage', zorder=9999, edgecolors='k',
#                            transform=ccrs.PlateCarree())
# point_uqam = ax.scatter(list_stat_lon, list_stat_lat, facecolor='orange', s=25, label='Climat sentinel station',
#                         zorder=9999, edgecolors='k', transform=ccrs.PlateCarree())

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=2)
# Save figure
fig.savefig(image_output_file, dpi=image_output_dpi, format='png',
            bbox_inches='tight')  # Most backends support png, pdf, ps, eps and svg
# os.system ('convert python.png ' + image_output_file)         # Convert python.png to 'image_output_file', i.e. .gif


# fig zommer
image_output_file = "/upslope/chalifour/projet_maitrise/fig/fig_zoom_domain.png"
# figsize = (5, 4.4)      # Figure size
fig = plt.figure()
# figsize=figsize
# Set projection defined by the cartopy object
ax = plt.axes(projection=ccrs.LambertConformal())

# Plotting - if needed
# --------------------

# Set corners of the maps
xll, yll = m.transform_point(lonE2p5[0, 0], latE2p5[0, 0], ccrs.PlateCarree())
xur, yur = m.transform_point(lonE2p5[-1, -1], latE2p5[-1, -1], ccrs.PlateCarree())

xm, ym = m.transform_point(-73.6053330201976, 45.52028815987258, ccrs.PlateCarree())

# Set geographic features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.5)  # couche ocean
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none')  # couche land
# ax.add_feature(cfeature.LAKES.with_scale('50m'))      # couche lac
ax.add_feature(cfeature.BORDERS.with_scale('50m'))  # couche frontieres
# ax.add_feature(cfeature.RIVERS.with_scale('50m'))     # couche rivières
coast = cfeature.NaturalEarthFeature(category='physical', scale='10m', facecolor='none',
                                     name='coastline')  # Couche côtières
ax.add_feature(coast, edgecolor='black')
ax.add_feature(cfeature.LAKES)
states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m',
                                                facecolor='none')  # Couche provinces
ax.add_feature(states_provinces, edgecolor='grey')

ax.set_extent([xll + 17, xur - 9, yll + 4, yur - 1], crs=m)

# To help the layout of the figure after saving
# fig.canvas.draw()
plt.tight_layout()  # To help with the layout of the figure after saving

# plot station
idx_del = df_disdro[df_disdro['Name'] == 'LAVAL'].index[0]
df_disdro.drop(index=idx_del, inplace=True)

# print(df_hydro_modif)
# idx_del_1 = df_hq[df_hq['Nom'] == 'Lac Bibitte (CM3Y)'].index[0]
# idx_del_2 = df_hq[df_hq['Nom'] == 'Lac Bibitte (CM3Y)'].index[0]
# new_df_hq = df_hq.drop(index=idx_del_1)
# for i,lon in enumerate(df_disdro['X'].values):
#     close = df_hq.iloc[(df_hq["XCoord"]-lon).abs().argsort()]
#     close = close.iloc[0]
#     # ax.text(close['XCoord'], close['YCoord'], f'{close["Nom"]}', transform=ccrs.PlateCarree(),fontsize=8)
#     if close["Nom"] !='Lac Bibitte (CM3Y)' and close["Nom"] !='Lac Pletipi (CMPI)' :
#         if i == 0:
#             point_stat_hq_0 = ax.scatter(close['XCoord'], close['YCoord'], facecolor='grey', s=55, marker='s',
#                                        label='HQ stations\nwith a disdrometer nearby',
#                                        zorder=9999, edgecolors='k', transform=ccrs.PlateCarree())
#         else:
#             point_stat_hq = ax.scatter(close['XCoord'], close['YCoord'], facecolor='grey', s=55, marker='s',
#                                        zorder=9999, edgecolors='k', transform=ccrs.PlateCarree())
#
#         idx_del = df_hq[df_hq['Nom'] == close['Nom']].index[0]
#         df_hq.drop(index=idx_del, inplace=True)

point_momo = ax.scatter(lon_momo, lat_momo, facecolor='red', s=120, marker="*", label='Forêt Montmorency', zorder=9999,
                        edgecolors='k', transform=ccrs.PlateCarree())
point_sondage = ax.scatter(lon_sond, lat_sond, facecolor='forestgreen', s=50, label='Atmospheric sounding\nstations',
                           zorder=9999, edgecolors='k',
                           transform=ccrs.PlateCarree())
point_uqam = ax.scatter(list_stat_lon, list_stat_lat, facecolor='peru', s=50, label='Climat sentinels',
                        zorder=999999, edgecolors='k', transform=ccrs.PlateCarree())
point_disdro = ax.scatter(df_disdro['X'].values, df_disdro['Y'].values, facecolor='grey', s=45, label='Disdrometers',
                          zorder=99999, edgecolors='k', transform=ccrs.PlateCarree(), marker="^")
point_stat_hq = ax.scatter(df_hq['XCoord'].values, df_hq['YCoord'].values, facecolor='royalblue', marker='s', s=55,
                           label='HQ stations',
                           zorder=999, edgecolors='k', transform=ccrs.PlateCarree())
data_path = sorted(glob.glob(fr"/upslope/chalifour/projet_maitrise/"))[0]
asos_stat = os.path.join(data_path, r"data_parcivel/asos_stati/asos_meta")
df_asos = pd.read_csv(asos_stat, header=0)
df_asos.set_index('stid', inplace=True)

for j, stat in enumerate(df_asos.index):
    lon_asos = df_asos['lon'][stat]
    lat_asos = df_asos['lat'][stat]

    if j == 0:
        scatter_iem = ax.scatter(lon_asos, lat_asos, s=15,
                                 zorder=999999, color='purple', edgecolors='k', transform=ccrs.PlateCarree(),
                                 label='IEM ASOS stat\n(if use)')

    else:
        ax.scatter(lon_asos, lat_asos, s=15,
                   zorder=999999, edgecolors='k', color='purple', transform=ccrs.PlateCarree())

legend_handle = [point_momo, point_sondage, point_uqam, point_disdro, point_stat_hq, scatter_iem]
ax.legend(handles=legend_handle, loc='upper center', bbox_to_anchor=(0.5, -0.01),
          fancybox=True, shadow=True, ncol=2, fontsize=12)
# Save figure

fig.savefig(image_output_file, dpi=image_output_dpi, format='png',
            bbox_inches='tight')  # Most backends support png, pdf, ps, eps and svg
# os.system ('convert python.png ' + image_output_file)         # Convert python.png to 'image_output_file', i.e. .gif
