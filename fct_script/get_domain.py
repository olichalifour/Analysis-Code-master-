



from typing import Dict
try:
    import rpnpy.librmn.all as rmn  # Module to read RPN files
    from rotated_lat_lon import RotatedLatLon  # Module to project field on native grid (created by Sasha Huziy)
except ImportError as err:
    print(f"RPNPY can only be use on the server. It can't be use on a personal computer."
          f"\nError throw :{err}")
import numpy as np
import cartopy.crs as ccrs  # Import cartopy ccrs



varname = 'TT'  # Name of variable to read
title = 'Domaines'


def get_domain_info(resolution:str) -> Dict:
    """

    :param resolution:
    :return:
    """


    dict_domain = {'corner':{},'domain_projm':{'lon':[],'lat':[]}}

    filename = '/chinook/roberge/Output/GEM5/Olivier/NAM-11m_ERA5_GEM50_PCPTYPEnil/Samples/NAM-11m_ERA5_GEM50_PCPTYPEnil_201509'  # Name of RPN file to read  # Name of RPN file to read
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
    tics = rmn.fstlir(fid, nomvar='^^', ip1=rec['ig1'], ip2=rec['ig2'],
                      ip3=rec['ig3'])  # Read corresponding tictac's

    # Close RPN input file
    rmn.fstcloseall(fid)  # Close the RPN file

    # 2-D Mapping - if needed
    # -----------------------
    # Get positions of rotated equator from IG1-4 of the tictac's
    (Grd_xlat1, Grd_xlon1, Grd_xlat2, Grd_xlon2) = rmn.cigaxg('E', tics['ig1'], tics['ig2'], tics['ig3'],
                                                              tics['ig4'])

    # Use Sasha's RotatedLatLon to get the rotation matrix
    rll = RotatedLatLon(lon1=Grd_xlon1, lat1=Grd_xlat1, lon2=Grd_xlon2,
                        lat2=Grd_xlat2)  # the params come from gemclim_settings.nml
    # Use Sasha's get_cartopy_projection_obj to get the cartopy object for the projection and domain defined by the coordinates
    m = rll.get_cartopy_projection_obj()

    if resolution == '12km':
        I = len(rect12kmlon)
        rect12kmx = np.zeros(I)
        rect12kmy = np.zeros(I)
        for i in range(I):
            rect12kmx[i], rect12kmy[i] = m.transform_point(rect12kmlon[i], rect12kmlat[i], ccrs.PlateCarree())

        xll, yll = m.transform_point(lon_12km[0, 0], lat_12km[0, 0], ccrs.PlateCarree())
        xur, yur = m.transform_point(lon_12km[-1, -1], lat_12km[-1, -1], ccrs.PlateCarree())

        rectx = rect12kmlon
        recty = rect12kmlat


    elif resolution == '2.5km':
        filename = '/pampa/roberge/Output/GEM5/Cascades_CORDEX/CLASS/Safe_versions/Spinup/ECan_2.5km_NAM11mP3_newP3_CLASS_DEEPoff_SHALon/Samples/ECan_2.5km_NAM11mP3_newP3_CLASS_DEEPoff_SHALon_201509'  # Name of RPN file to read
      # Name of RPN file to read
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
        xll, yll = m.transform_point(lonE2p5[0, 0], latE2p5[0, 0], ccrs.PlateCarree())
        xur, yur = m.transform_point(lonE2p5[-1, -1], latE2p5[-1, -1], ccrs.PlateCarree())
        rectx = rectE2p5x
        recty = rectE2p5y

    else:
        raise ValueError('Resolution is 2.5km or 12km in the moment. Choose one of these two')

    dict_domain['corner'] = {'xll': xll, 'yll': yll, 'xur': xur, 'yur': yur}
    dict_domain['domain_projm'] = {'lon':rectx,'lat':recty}
    dict_domain['projection'] = m

    return dict_domain