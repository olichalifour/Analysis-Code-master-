#
# Before opening Python execute:
#   . s.ssmuse.dot rpnpy



from datetime import datetime, timedelta

from urllib.request import urlopen
try:
    import rpnpy.librmn.all as rmn
except ImportError as err:
    print(f"RPNPY can only be use on the server. It can't be use on a personal computer."
          f"\nError throw :{err}")

#from rotated_lat_lon import RotatedLatLon
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
#import rpn_functions
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path
import matplotlib.patches as patches




# def main():
#     output_model_points(outfname = 'station_pts_wcps_hrdps_t2m.csv' )

###################################################################
        ####### Checking stations inside domain#######
###################################################################

def station_inside_domain(obs_p, station_info, croping_list, pm_file):
    [lon_s0,lon_s1,lat_s0,lat_s1] = croping_list
    fid = rmn.fstopenall(pm_file, rmn.FST_RO)
    rec = rmn.fstlir(fid, nomvar='PR')
    mat_PR = rec['d'][lon_s0:lon_s1,lat_s0:lat_s1]

    m, lon, lat = m_from_tic(fid, rec, croping_list)
    xx, yy = m(lon,lat)
    xxC_q =np.zeros(len(station_info.index))
    yyC_q =np.zeros(len(station_info.index))
    
    xxC = list()
    yyC = list()  
    cnt_i = 0

    index_list = [datetime.strptime(obs_p.index[i], '%Y-%m-%d')  for i in range(len(obs_p.index))]
    time_list = pd.date_range(np.min(index_list), np.max(index_list))
    
    obs_p_r = pd.DataFrame()
    station_info_r = pd.DataFrame()
    for cnt_i in  range(len(station_info.index)):
        xxC_q, yyC_q = m(float(station_info.iloc[cnt_i].longitude)+360, float(station_info.iloc[cnt_i].latitude) )

        if (xxC_q >= np.min(np.min(xx)) ) & (xxC_q <= np.max(np.max(xx)) ) & (yyC_q >= np.min(np.min(yy)) ) & (yyC_q <= np.max(np.max(yy)) ) :
            station_info_r = station_info_r.append( station_info.iloc[cnt_i] )

            for ii in range(len(time_list)):

                
                obs_p_r = obs_p_r.append( obs_p[obs_p.index ==  str(time_list[ii])[0:10]].iloc[cnt_i] )
            
            
            xxC.append(xxC_q)
            yyC.append(yyC_q)

    return obs_p_r, station_info_r, xxC, yyC

def open_var_2d(fname, var, ip1=None, ip2=None, datev=None):

    '''
    Get variable information for 2D grid. Return variable and lat/lon grids.

    '''

    fid = rmn.fstopenall(fname,rmn.FST_RO)   # Open the file

    #datev needs to be used for CaPA files which contain multiple times per file

    if datev is not None:
        key1 = rmn.fstinf(fid, nomvar=var)
        rec = rmn.fstlirx(key1, fid, nomvar=var, datev=datev)
    elif (ip1 is not None) and (ip2 is not None):
        rec = rmn.fstlir(fid,nomvar=var, ip1=ip1, ip2=ip2)        # Read the full record of variable 'varname'
    elif (ip1 is not None):
        rec = rmn.fstlir(fid,nomvar=var, ip1=ip1)
    elif ip2 is not None:
        rec = rmn.fstlir(fid,nomvar=var, ip2=ip2)
    else:
        rec = rmn.fstlir(fid,nomvar=var)


    var_data = rec['d']                            # Assign 'field' to the data of 'varname'
    mygrid = rmn.readGrid(fid,rec)              # Get the grid information for the (LAM) Grid -- Reads the tictac's
    latlondict = rmn.gdll(mygrid)               # Create 2-D lat and lon fields from the grid information
    lat = latlondict['lat']                     # Assign 'lat' to 2-D latitude field
    lon = latlondict['lon']                     # Assign 'lon' to 2-D longitude field
    rmn.fstcloseall(fid)                        # Close the RPN file
    return var_data, lat, lon


def open_var_one_pt_interp(fname, var, lat_pt, lon_pt, ip1=None, datev=None):


    '''
    Get variable information at lat/lon point by interpolation of nearby gridpoints.
    otherwise find nearest gridpoint to lat_pt, lon_pt
    '''   


    #If u or v, use function to calculate vector at point. Otherwise scalar.


    if var in ['UU','VV']:

        #Split up fname to get u and v winds

        # fname_u = fname[:-17]+'UU'+fname[-15:]
        # fname_v = fname[:-17]+'VV'+fname[-15:]



        list_path_UU = fname.split('/')
        list_path_UU.insert(6, "Rotated_Wind_Vectors")
        list_path_UU[-1] = f"UU_{list_path_UU[-1]}"
        path_UU = "/".join(list_path_UU)

        list_path_VV = fname.split('/')
        list_path_VV.insert(6, "Rotated_Wind_Vectors")
        list_path_VV[-1] = f"VV_{list_path_VV[-1]}"
        path_VV = "/".join(list_path_VV)

        fid_u = rmn.fstopenall(path_UU,rmn.FST_RO)   # Open the file
        fid_v = rmn.fstopenall(path_VV,rmn.FST_RO)   # Open the file
        if ip1 is None:
            urec = rmn.fstlir(fid_u, nomvar='UU')
            vrec = rmn.fstlir(fid_v, nomvar='VV')
        else:
            urec = rmn.fstlir(fid_u, nomvar='UU', ip1=ip1)
            vrec = rmn.fstlir(fid_v, nomvar='VV', ip1=ip1)

        mygrid = rmn.readGrid(fid_u,urec)              # Get the grid information for the (LAM) Grid -- Reads the tictac's
        rmn.fstcloseall(fid_u)                        # Close the RPN file
        rmn.fstcloseall(fid_v)
        var_pt = rmn.gdllvval(mygrid, [lat_pt], [lon_pt], urec['d'],vrec['d'])  #Get value of var interpolated to lat/lon point
        # print(var_pt)
        # if var == 'UU':
        #     var_pt = var_pt[0]
        # else:
        #     var_pt = var_pt[1]
    else:

        fid = rmn.fstopenall(fname,rmn.FST_RO)   # Open the file
        if datev is not None:
            key1 = rmn.fstinf(fid, nomvar=var)
            rec = rmn.fstlirx(key1, fid, nomvar=var, datev=datev)
        elif ip1 == None:
            rec = rmn.fstlir(fid,nomvar=var)    
        else:
            rec = rmn.fstlir(fid,nomvar=var,  ip1=ip1)    # Read the full record of variable 'varname'

        mygrid = rmn.readGrid(fid,rec)              # Get the grid information for the (LAM) Grid -- Reads the tictac's
        rmn.fstcloseall(fid)                        # Close the RPN file
        var_pt = rmn.gdllsval(mygrid, [lat_pt], [lon_pt], rec['d']) #Get value of var interpolated to lat/lon point

    return var_pt

def wind_direction(u, v):
    r"""Taken from MetPy
    Compute the wind direction from u and v-components.

    Parameters
    ----------
    u : `pint.Quantity`
        Wind component in the X (East-West) direction
    v : `pint.Quantity`
        Wind component in the Y (North-South) direction

    convention : str
        Convention to return direction; 'from' returns the direction the wind is coming from
        (meteorological convention), 'to' returns the direction the wind is going towards
        (oceanographic convention), default is 'from'.

    Returns
    -------
    direction: `pint.Quantity`
        The direction of the wind in intervals [0, 360] degrees, with 360 being North,
        direction defined by the convention kwarg.

    See Also
    --------
    wind_components

    Notes
    -----
    In the case of calm winds (where `u` and `v` are zero), this function returns a direction
    of 0.
    """

    wdir = 90. - np.arctan2(-v, -u)*(180/np.pi)
    origshape = wdir.shape
    wdir = np.atleast_1d(wdir)
    mask = wdir <= 0
    if np.any(mask):
        wdir[mask] += 360.

    # avoid unintended modification of `pint.Quantity` by direct use of magnitude
    calm_mask = (np.asarray(u) == 0.) & (np.asarray(v) == 0.)
    # np.any check required for legacy numpy which treats 0-d False boolean index as zero

    if np.any(calm_mask):
        wdir[calm_mask] = 0.
    return wdir.reshape(origshape)

def lambert_map(extent=(-82, -75, 41, 46), cent_lon =-80,figsize=(14, 12), fig = None, ax = None):
    '''
    Lambert Conformal map
    '''
    proj = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=35,
                                 standard_parallels=[35])
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=proj)

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',  name='admin_1_states_provinces_lines',
        scale='50m', facecolor='none')
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor='0.9')
    lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                            edgecolor='None',
                                            facecolor=[(0.59375 , 0.71484375, 0.8828125)])
    lakes_50m_edge= cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                            edgecolor='gray',
                                            facecolor='None')
    #(0.59375 , 0.71484375, 0.8828125)
    ax.add_feature(land_50m); 
    ax.add_feature(lakes_50m, zorder=3); 
    ax.add_feature(lakes_50m_edge, zorder=10); 
    #ax.add_feature(cfeature.LAKES, edgecolor='white', zorder=10);
    ax.add_feature(cfeature.BORDERS, zorder=10); 
    ax.add_feature(states_provinces, edgecolor='gray', zorder=10)
    ax.coastlines('50m', zorder=10)

    # Set plot bounds
    ax.set_extent(extent)
    return fig, ax


def plateCarree(figsize=11):
    '''
    Basic plateCarree map
    '''
    proj = ccrs.PlateCarree(central_longitude=-95)
    fig, ax = plt.subplots(figsize=(figsize,figsize), subplot_kw=dict(projection=proj), facecolor=facecolor)
    ax.set_extent([-140, -50, 21, 51])               
   
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',  name='admin_1_states_provinces_lines',
        scale='50m', facecolor='none')
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor='0.9')

    ax.add_feature(land_50m); 
    ax.add_feature(cfeature.LAKES, edgecolor='white');
    ax.add_feature(cfeature.BORDERS, zorder=10); 
    ax.add_feature(states_provinces, edgecolor='gray', zorder=10)
    ax.coastlines('50m', zorder=10)
    '''
    Uncomment to plot lat/lon lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True, 
                              linewidth=2, color='gray', alpha=0.5, linestyle='--', zorder=50)
    gl.xlocator = mticker.FixedLocator(np.arange(-180,190,10))    
    gl.ylocator = mticker.FixedLocator(np.arange(0,90,10))
    gl.ylabel_style = {'fontsize':14};  gl.xlabel_style = {'fontsize':14};
    gl.xlabels_top = False
    gl.ylabels_right = False
    '''
    return fig, ax

def regrid_to_rdps(f_rdpsa, f_sim, var, ip1_rdpsa, ip1_sim):
    '''
    Read the RDPS analysis grid for a given field and iterpolate the simulation's data for the same field to the
    RDPS grid. Returns rdps data, regridded data and latitude and longitude of new grid.
    '''
    fid = rmn.fstopenall(f_rdpsa,rmn.FST_RO)   # Open the file
    rec_rdps = rmn.fstlir(fid,nomvar=var, ip1=ip1_rdpsa)        # Read the full record of variable 'varname'

    rdps_grid = rmn.readGrid(fid,rec_rdps)              # Get the grid information for the (LAM) Grid -- Reads the tictac's
    latlondict = rmn.gdll(rdps_grid)               # Create 2-D lat and lon fields from the grid information
    lat_rdps = latlondict['lat']                     # Assign 'lat' to 2-D latitude field
    lon_rdps = latlondict['lon']          
    rmn.fstcloseall(fid)                        # Close the RPN file
    
    #Open field for simulation to verify    
    fid = rmn.fstopenall(f_sim,rmn.FST_RO)   # Open the file
    rec_to_regrid = rmn.fstlir(fid,nomvar=var, ip1=ip1_sim)        # Read the full record of variable 'varname'
    grid_to_regrid = rmn.readGrid(fid,rec_to_regrid)              # Get the grid information for the (LAM) Grid -- Reads the tictac's
    rmn.fstcloseall(fid)                        # Close the RPN file
    
    #Interpolate
    gridsetid = rmn.ezdefset(rdps_grid, rec_to_regrid)
    rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_CUBIC)
    rec_regridded = rmn.ezsint(rdps_grid['id'], rec_to_regrid['id'], rec_to_regrid['d'])
    
    return rec_rdps['d'], rec_regridded, lat_rdps, lon_rdps

def multipanel_map(nrows, ncols,extent=(-82, -75, 41, 46), cent_lon = -80, figsize=(15,8),  box=None, hspace=-0.6,dpi=200,):
    '''
    Plot a multipanel figure with maps in nrows 
    '''
    proj = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=35,
                                 standard_parallels=[35])
    zoom_proj = ccrs.PlateCarree(central_longitude=0)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,  subplot_kw=dict(projection=proj), dpi=dpi)
    
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor='0.9')
    lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                            edgecolor='None',
                                            facecolor=[(0.59375 , 0.71484375, 0.8828125)])
    lakes_50m_edge= cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                            edgecolor='gray',
                                            facecolor='None')
    states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',  name='admin_1_states_provinces_lines',
                scale='50m', facecolor='none')
    
    for ax in axs:
        if len(axs.shape)==1:        
            naxes = 1
            ax = [ax]
        else:
            naxes = len(ax)
        for nax in np.arange(0,naxes):
            ax[nax].outline_patch.set_linewidth(0.5)
            ax[nax].set_extent(extent,zoom_proj)  
            #ax[1].set_extent(extent,ccrs.PlateCarree())  
            ax[nax].coastlines('50m', zorder=10, linewidths=0.5)        
            

            ax[nax].add_feature(land_50m)      
            ax[nax].add_feature(lakes_50m, zorder=3); 
            ax[nax].add_feature(lakes_50m_edge, zorder=10); 
            ax[nax].add_feature(cfeature.BORDERS, zorder=10, linewidths=0.5); 
            ax[nax].add_feature(states_provinces, edgecolor='0.5', zorder=9, linewidths=0.5)
                    
            #Plot box        
            if box is not None:
                verts = [(box['west'], box['south']), (box['west'], box['north']), (box['east'], box['north']), (box['east'], box['south']), (0.,0.)]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='none', ec='black', lw=2, transform=ccrs.PlateCarree(),
                                          ls='--', zorder=99999999)
                ax[nax].add_patch(patch)
            ax[nax].set_extent(extent)
    fig.subplots_adjust(wspace=0.05, hspace=hspace)
    return fig, axs

def rmse_n(predictions, targets):
    '''Calcuate root mean square error '''
    return np.sqrt(np.nanmean((predictions - targets) ** 2))/(np.nanmax(targets)-np.nanmin(targets))

def rmse(predictions, targets):
    '''Calcuate root mean square error '''
    return np.sqrt(np.nanmean((predictions - targets) ** 2))
def mse(predictions, targets):
    '''Calcuate mean square error '''
    return np.nanmean((predictions - targets) ** 2)

def lin_interp(z_stn, z_top, z_bot, var_top, var_bot):
    '''
    Linearly interpolate to find value of var at station point
    between z_top and z_bot
    ''' 
    slope = (z_top-z_bot)/(var_top-var_bot)
    var_stn = var_bot + (z_stn-z_bot)/slope
    return(var_stn)

def date_to_datev(date):

    '''
    Convert a datetime object to a datev code for RPN
    '''

    datev_0 = 438776000  # 2020-01-01 00
    date_0 = pd.Timestamp('2020-01-01 00')
    date_diff = date - date_0
    six_hr_periods = date_diff.total_seconds() / 3600 / 6
    datev = int(datev_0 + six_hr_periods * 5400)
    return datev

def download_sounding_data(stn,date, output_dir='/upslope/chalifour/Projet_ete_2021/sounding/'):
    '''
    Function to download University of Wyoming sounding data and output it to a txt file
    Modified from https://kbkb-wx-python.blogspot.com/2015/07/plotting-sounding-data-from-university.html
    Input station code, date (as pd.Timestamp object), and output directory
    Outputs txt file in output_dir that can be read using pandas (pd.read_csv(sounding_fname, delim_whitespace=True))
    Sometimes University of Wyoming website is overloaded and will refuse requests, need to run function a few seconds
    later if this happens
    '''
    year=date.strftime('%Y')
    month=date.strftime('%m')
    day = date.strftime("%d")
    hour = date.strftime('%H')

    # 1)
    # Wyoming URL to download Sounding from
    url = 'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR='+year+'&MONTH='+month+'&FROM='+day+hour+'&TO='+day+hour+'&STNM='+stn

    content = urlopen(url).read().decode('utf-8')

    # 2)
    # Remove the html tags
    import re

    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext

    data_text = cleanhtml(content)


    # 3)
    # Split the content by new line.
    header = data_text.split("\n",data_text.count("\n"))[6]+'\n'
    splitted = data_text.split("\n",data_text.count("\n"))[9:-59]


    # 4)
    # Write this splitted text to a .txt document
    Sounding_filename = output_dir+'/'+str(stn)+'.'+str(year)+str(month)+str(day)+str(hour)+'.txt'
    f = open(Sounding_filename,'w')
    f.write(header)
    for line in splitted[:]:
        if re.search('                                                              ',line):
            pass
        else:
            f.write(line+'\n')
    f.close()

def dt64_to_datetime(datetime_obj_list):
    dt_list = list()
    for i in range(len(datetime_obj_list)):
        ts = (datetime_obj_list[i] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        dt_list.append(datetime.utcfromtimestamp(ts))
    return np.array(dt_list)

def get_time_height(nc):
    time   = np.array(dt64_to_datetime(nc.time))
    dt     = time[1]-time[0]
    height =  nc.range.data/ 1000 # meters to kilometers

    # pas mal sure que ya un offset de 1.4km
    # height = nc.range.data / 1000 - 1.4  # meters to kilometers

    dh     = height[1]-height[0]

    return time,dt,height,dh

def W_plot(nc, ax, fs=12):
    time, dt, height, dh = get_time_height(nc)

    '''
    Colormap construction
    '''

    nb_couleur = 256
    level = np.linspace(-1, 8, nb_couleur)

    cmap1 = np.array([np.linspace(mcolors.to_rgba('r')[i], mcolors.to_rgba('lightgrey')[i], 10) for i in range(4)]).T
    cmap2 = np.array([np.linspace(mcolors.to_rgba('lightgrey')[i], mcolors.to_rgba('b')[i], 20) for i in range(4)]).T
    cmap3 = np.array([np.linspace(mcolors.to_rgba('b')[i], mcolors.to_rgba('g')[i], 20) for i in range(4)]).T
    cmap4 = np.array([np.linspace(mcolors.to_rgba('g')[i], mcolors.to_rgba('y')[i], 20) for i in range(4)]).T
    cmap4 = np.array([np.linspace(mcolors.to_rgba('y')[i], mcolors.to_rgba('C1')[i], 20) for i in range(4)]).T
    cmap_12 = np.concatenate([cmap1, cmap2, cmap3, cmap4])

    cmap_brg = mcolors.LinearSegmentedColormap.from_list('', cmap_12, N=1000)

    '''
    Plot and colorbar
    '''
    cmesh = ax.pcolormesh(time - dt / 2, height - dh / 2, nc.VEL.T, cmap=cmap_brg, vmin=-1, vmax=8, shading='auto')
    ax.set_ylim([ height[0] , height[-1] ])
    ax.set_xlim([time[0],time[-1]])
    return ax,cmesh


def ze_plot(nc, ax, fs=12):
    time, dt, height, dh = get_time_height(nc)

    '''
    Cmap construction
    '''

    env_can = ['#98CCFE', '#0099FE', '#00FF65', '#00CC02', '#009902' \
        , '#006601', '#FEFF34', '#FFCC00', '#FF9900', '#FF6600', \
               '#FF0000', '#FE0399', '#9934CC', '#660199']

    env_can_ar = np.linspace(0, 45, 100)
    cmap_fr = mcolors.LinearSegmentedColormap.from_list('', env_can, N=1000)
    norm = mcolors.BoundaryNorm(boundaries=env_can_ar, ncolors=1000)

    '''
    Plot and colorbar
    '''
    cmesh = ax.pcolormesh(time - dt / 2, height - dh / 2, nc.Ze.T, cmap=cmap_fr, vmin=0, vmax=45,shading='auto')
    ax.set_ylim([height[0], height[-1]])
    ax.set_xlim([time[0], time[-1]])
    return ax,cmesh

