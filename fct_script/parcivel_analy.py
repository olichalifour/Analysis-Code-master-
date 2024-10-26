
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
import os
from datetime import datetime, timedelta


start = pd.date_range(start='2021-12-6 00',end='2021-12-7 19',freq='2H')


end = pd.date_range(start='2021-12-6 02',end='2021-12-7 20',freq='2H')

# ______________________________________________________________________________
#           C O N S T A N T S :  P R E C I P I T A T I O N   T Y P E S
# Rain, Graupel values from Ishizaka 2013
# Wet, Dry values from Rasmussen 1998
# [mass]=mg [diameter]=mm [velocity]=m/s
pType_ch = ['Rain', 'Graupel', 'Wet snow', 'Dry snow']  #

# EMPIRICAL PRECIPITATION CURVES: V(D)= a(cD^b) = v_a*((v_c*D)**v_b)
#           rain=0,      graupel=1, wetsnow=2,          drysnow=3
v_a = [3.78, 1.3, 2.14, 1.07]  # V(D) coefficient
v_b = [0.67, 0.66, 0.20, 0.20]  # V(D) exponent
v_c = [1., 1.0, 0.1, 0.1]  # V(D) units adjustments
# EMPIRICAL PRECIPITATION CURVES: m(D)=a(cD^b) = m_a*((m_c*D)**m_b)
m_a = [0.52, 0.078, (3.14 / 6.) * 0.072 * 1e3, (3.14 / 6.) * 0.017 * 1e3]  # m(D) coefficient
m_b = [3.0, 2.8, -1. + 3., 3. - 1.]  # m(D) exponent
m_c = [1., 1.0, 0.1, 0.1]  # m(D) units adjustments


# m_a =       [0.52,       (3.14/6.)*0.072*1e3,      (3.14/6.)*0.072*1e3,   (3.14/6.)*0.017*1e3]  # m(D) coefficient
# m_b =       [3.0,        -1.+3.,        -1.+3.,            3.-1.]            # m(D) exponent
# m_c =       [1.,         0.1,         0.1,             0.1]             # m(D) units adjustments

def v_fct(i, D):  # i is the index of the precipitation type to get corresponding coeffs.
    return (v_a[i] * ((v_c[i] * D) ** v_b[i]))


def m_fct(i, D):  # i is the index of the precipitation type to get corresponding coeffs.
    return (m_a[i] * ((m_c[i] * D) ** m_b[i]))


project_path = os.path.abspath(os.path.join(__file__, "../../.."))

folder_data_path = os.path.join(project_path, r"master_degree/data_parcivel/site_neige/raw")

# %%
folders = os.listdir(folder_data_path)

# %%
file_paths = []
for folder in folders:
    # files = os.listdir(path + folder)
    # files = [f for f in files if f[-3:] == 'txt']
    file_paths.append(folder_data_path+'/'+ folder)

# files = [f for file in file_paths for f in file]
# # %%
# print(files)
df = []
for f in file_paths:

    df.append(pd.read_csv(f,header=1,index_col=0,skiprows=[2,3]))

df = pd.concat(df)

df.index = pd.to_datetime(df.index, infer_datetime_format=True)
df.sort_index(inplace=True)



def creat_index_VD(n_bin_v:float,n_bin_d:float)->list:
    list_index=[]
    for d in range(n_bin_d):
        for v in range(n_bin_v):
            list_index.append(f'V{v}D{d}')

    return list_index

list_index_vd = creat_index_VD(32,32)

df.columns = ['Intensity of precipitation', 'Radar reflectivity', 'Number of detected particles',
             'Heating current', 'Sensor voltage', 'Snow intensity','Precipitation since start',
             'Weather code SYNOP WaWa', 'Weather code METAR/SPECI', 'Weather code NWS', 'MOR Visibility',
              'Signal amplitude of Laserband', 'Temperature in sensor', 'Kinetic Energy' ] + list(df.columns[14:77])+list_index_vd

#%%
Parsivel_prec = df[['Intensity of precipitation', 'Radar reflectivity', 'Snow intensity','Heating current', 'Sensor voltage','Signal amplitude of Laserband',
       'Temperature in sensor','Kinetic Energy']]


#%%
Parsivel_prec_60min = Parsivel_prec.resample('60T', closed  = 'right', label = 'right').mean()

#%%
Parsivel_pcpn_type = df[['Weather code SYNOP WaWa', 'Weather code METAR/SPECI', 'Weather code NWS']]
#%%
dMeanFull = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, \
        1.875, 2.125, 2.375, 2.75, 3.25, 3.75, 4.25, 4.75, 5.5, 6.5, 7.5, 8.5, \
        9.5, 11, 13, 15, 17, 19, 21.5, 24.5]


vMeanFull = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, \
         1.7, 1.9, 2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, \
         15.2, 17.6, 20.8]


#%%
dMean = dMeanFull[2:] # 2 first classes outside of the range of the OTT Parsivel 2
vMean = vMeanFull[2:] # 2 first classes outside of the range of the OTT Parsivel 2

dMean = np.array(dMean)
vMean = np.array(vMean)
#%%
deltaD = np.array([ 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, \
         0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, \
         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])


deltaV = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.2, 0.2, 0.2, 0.2, 0.2,
         0.4, 0.4, 0.4, 0.4, 0.4,
         0.8, 0.8, 0.8, 0.8, 0.8,
         1.6, 1.6, 1.6, 1.6, 1.6,
         3.2, 3.2])


#%%
# Take only the columns with velocity and diameter data
cols = [c for c in df.columns if c[0]=='V']

#%%
# Take the first 2 columns of velocity (D0 and D1) out
colsDiameter = [c for c in cols if int(c.split('D')[-1])>=2]

#%%
# Take the first 2 columns of velocity (V0 and V1) out
colsDiameterVel = [c for c in colsDiameter if int(c.split('D')[0][1:])>1]

#%%
Parsivel = df[colsDiameterVel]
#%%
Parsivel = Parsivel.replace(np.nan, 0)
#%%
Parsivel_1h = Parsivel.resample('60T', label = 'right', closed = 'right').sum()

for i in range(len(start)):
    plt.figure(figsize=(15, 8), facecolor='white')

    pType_color = ['m', 'tab:green', 'tab:orange', 'tab:blue', ]  # ['firebrick','lime',    'm',       'orange']#
    cmap = mcolors.ListedColormap(pType_color)

    d = np.array(dMeanFull)
    plt.plot(d, v_fct(0, d), label=pType_ch[0], lw=3, color=pType_color[0])
    plt.plot(d, v_fct(1, d), label=pType_ch[1], lw=3, color=pType_color[1])
    plt.plot(d, v_fct(2, d), label=pType_ch[2], lw=3, color=pType_color[2])
    plt.plot(d, v_fct(3, d), label=pType_ch[3], lw=3, color=pType_color[3])

    mat = Parsivel.loc[start[i]:end[i]].sum(axis=0).values.reshape(len(vMean), len(dMean))

    #
    print(mat)
    # mat[mat == 0] = np.nan

    X, Y = np.meshgrid(dMean - deltaD / 2, vMean - deltaV / 2)

    plt.pcolormesh(X, Y, mat, cmap='jet', norm=colors.LogNorm(vmin=1, vmax=1e4),shading='auto')

    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xlabel('Diameter [mm]', fontsize=25)
    plt.ylabel('Fallspeed [m s$^{-1}$]', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    cbar = plt.colorbar(extend='max')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('# of particles', fontsize=20)

    plt.title(f'{start[i].strftime("%Y/%m/%d %H%M")} to {end[i].strftime("%Y/%m/%d %H%M")} ', fontsize=20)

    plt.savefig(f'/Users/olivier1/Documents/GitHub/master_degree/fig/parcivel_exemple/exemple_parcivel_{i}.png', bbox_inches='tight')
    plt.close()