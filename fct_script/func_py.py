import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
import cartopy.crs as ccrs  # Import cartopy ccrs
import cartopy.feature as cfeature  # Import cartopy common features
try:
    import rpnpy.librmn.all as rmn  # Module to read RPN files
    from rotated_lat_lon import RotatedLatLon  # Module to project field on native grid (created by Sasha Huziy)
except ImportError as err:
    print(f"RPNPY can only be use on the server. It can't be use on a personal computer."
          f"\nError throw :{err}")



def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):

    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = colors.ListedColormap(cols)
    return cmap


def get_colormap_precip():
    cmap = cm.get_cmap('jet', 256)
    cmap = cmap(np.linspace(0.2, 1, 256))[::12]
    cmap[0, -1] = 0
    # cmap=np.insert(cmap,0,[1 ,1, 1, 0],axis=0)
    bounds_1 = np.linspace(0, 120., 19)
    bounds_1[0] = 1
    # bounds = np.insert(bounds,0,0)
    # bounds = [0,1.5,5,10,20,40,60,70,90,110]
    cmap = ListedColormap(cmap)
    norm = BoundaryNorm(bounds_1, cmap.N)
    cmap.set_over('black')
    return cmap


def get_proj_extent():
    filename = '/chinook/roberge/Output/GEM5/Olivier/NAM-11m_ERA5_GEM50_PCPTYPEnil/Samples/NAM-11m_ERA5_GEM50_PCPTYPEnil_202010'  # Name of RPN file to read  # Name of RPN file to read
    varname = 'TT'  # Name of variable to read
    title = 'Domaines'
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

    ##################################################################################################################################
    filename = '/pampa/roberge/Output/GEM5/Cascades_CORDEX/CLASS/Safe_versions/Spinup/ECan_2.5km_NAM11mP3_newP3_CLASS_DEEPoff_SHALon/Samples/ECan_2.5km_NAM11mP3_newP3_CLASS_DEEPoff_SHALon_201509'  # Name of RPN file to read
    fid = rmn.fstopenall(filename, rmn.FST_RO)  # Open the file
    rec = rmn.fstlir(fid, nomvar=varname)  # Read the full record of variable 'varname'
    field = rec['d']  # Assign 'field' to the data of 'varname'
    # Read 2-D latitudes & longitudes - if needed
    mygrid = rmn.readGrid(fid, rec)  # Get the grid information for the (LAM) Grid -- Reads the tictac's
    latlondict = rmn.gdll(mygrid)  # Create 2-D lat and lon fields from the grid information
    latE2p5 = latlondict['lat']  # Assign 'lat' to 2-D latitude field
    lonE2p5 = latlondict['lon']  # Assign 'lon' to 2-D longitude field
    rmn.fstcloseall(fid)  # Close the RPN file
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
    return m,lonE2p5,latE2p5




