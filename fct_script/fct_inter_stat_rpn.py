
try:
    import rpnpy.librmn.all as rmn  # Module to read RPN files
    from rotated_lat_lon import RotatedLatLon  # Module to project field on native grid (created by Sasha Huziy)
except ImportError as err:
    print(f"RPNPY can only be use on the server. It can't be use on a personal computer."
          f"\nError throw :{err}")


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