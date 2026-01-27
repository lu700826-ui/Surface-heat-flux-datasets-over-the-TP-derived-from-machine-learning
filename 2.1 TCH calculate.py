

import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from GTCH import cal_uct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import proplot as pplt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib_scalebar.scalebar import ScaleBar

def funclip(data,datafused):
    data=data.sel(time=slice('2000/01/01','2021/12/31'))
    data = data.rio.write_crs("EPSG:4326")
    datafused = datafused.rio.write_crs("EPSG:4326")
    data= data.rio.reproject_match(datafused)
    return  data

def extract_seasonal_data(data, season_name):
    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }

    months = season_months[season_name]
    seasonal_data = data.sel(time=data.time.dt.month.isin(months))

    return seasonal_data

hh=['LH','SH']
nn=['E','H']
kk=1
v_a=[['slhf','LHTFL_GDS4_SFC_S130','EFLUX','LH'],['sshf','SHTFL_GDS4_SFC_S130','HFLUX','SH']]
ERA5_LH= xr.open_dataset(rf'F:\LHSH\original_month_data\monthly\Monthly_ERA5_{hh[kk]}_W_M2_0909.nc',engine='netcdf4', mode='r')[v_a[kk][0]]
JRA55_LH= xr.open_dataset(rf'F:\LHSH\original_month_data\monthly\Monthly_JRA55_{hh[kk]}_W_M2_0909.nc',engine='netcdf4', mode='r')[v_a[kk][1]]
MERRA2_LH= xr.open_dataset(rf'F:\LHSH\original_month_data\monthly\Monthly_MERRA2_{hh[kk]}_W_M2_0909.nc',engine='netcdf4', mode='r')[v_a[kk][2]]
CRA_LH= xr.open_dataset(rf'F:\LHSH\original_month_data\monthly\Monthly_CRA_{hh[kk]}_W_M2_0909.nc', engine='netcdf4',mode='r')[v_a[kk][3]]
if kk==0:
    Datafused_LH= xr.open_dataset(rf'F:\LHSH\XLUcode\Xgboost0701\CatBoost_final_{hh[kk]}_predictions_0250915.nc',engine='netcdf4', mode='r')['prediction']
else:
    Datafused_LH= xr.open_dataset(rf'F:\LHSH\XLUcode\Xgboost0701\CatBoost_final_{hh[kk]}_predictions_0250925.nc',engine='netcdf4', mode='r')['prediction']
    Datafused_LH= Datafused_LH*1.5
monthly_mean = Datafused_LH.resample(time='1M').mean()


ERA5_LH = funclip(ERA5_LH,monthly_mean)
JRA55_LH = funclip(JRA55_LH,monthly_mean)
MERRA2_LH = funclip(MERRA2_LH,monthly_mean)
CRA_LH= funclip(CRA_LH,monthly_mean)
Datafused_LH=monthly_mean

y_size = len(CRA_LH.y)
x_size = len(CRA_LH.x)
num_datasets = 5  #
uct_results = np.full((y_size, x_size, num_datasets), np.nan)
r_uct_results = np.full((y_size, x_size, num_datasets), np.nan)
# calculate TCH result
for i in range(1,len(CRA_LH.y)):
    for j in range(1,len(CRA_LH.x)):
        cra_ts=CRA_LH[:,i,j].values
        era5_ts= ERA5_LH[:,i,j].values
        merra2_ts= MERRA2_LH[:,i,j].values
        jra55_ts=JRA55_LH[:,i,j].values
        datafused_ts=Datafused_LH[:,i,j].values
        combined = np.column_stack([cra_ts, era5_ts, merra2_ts, jra55_ts,datafused_ts])
        uct, r_uct=cal_uct(combined)
        uct_results[i, j, :] = uct
        r_uct_results[i, j, :] = r_uct

        print('finish01')

dataset_names = ['CRA', 'ERA5', 'MERRA2', 'JRA55', 'Fused']

# create xarray Dataset
result_ds = xr.Dataset({
    'absolute_uncertainty': (['y', 'x', 'dataset'], uct_results),
    'relative_uncertainty': (['y', 'x', 'dataset'], r_uct_results)
}, coords={
    'y': CRA_LH.y.values,
    'x': CRA_LH.x.values,
    'dataset': dataset_names
},
attrs={
    'description': 'Triple Collocation Hardware (TCH) uncertainty analysis results',
    'units': {
        'absolute_uncertainty': 'same as input data units',
        'relative_uncertainty': 'percentage (%)'
    },
    'data_sources': 'GLEAM, CRA, ERA5, MERRA2, JRA55, Fused'})

result_ds.absolute_uncertainty.attrs = {
    'long_name': 'Absolute Uncertainty',
    'units': 'same as input data'}
result_ds.relative_uncertainty.attrs = {
    'long_name': 'Relative Uncertainty',
    'units': 'percent'}
result_ds.dataset.attrs = {
    'long_name': 'Data source'}

result_ds_fixed = xr.Dataset({
    'absolute_uncertainty': (['y', 'x', 'dataset'], result_ds.absolute_uncertainty.values),
    'relative_uncertainty': (['y', 'x', 'dataset'], result_ds.relative_uncertainty.values)
},
coords={
    'y': result_ds.y.values,
    'x': result_ds.x.values,
    'dataset': result_ds.dataset.values
})

result_ds_fixed.attrs = {
    'description': 'Triple Collocation Hardware (TCH) uncertainty analysis results',
    'data_sources': 'GLEAM, CRA, ERA5, MERRA2, JRA55, Fused',
    'creation_date': '2024-09-24'
}

result_ds_fixed.absolute_uncertainty.attrs = {
    'long_name': 'Absolute Uncertainty',
    'units': 'same as input data units'
}

result_ds_fixed.relative_uncertainty.attrs = {
    'long_name': 'Relative Uncertainty',
    'units': 'percentage (%)'
}

result_ds_fixed.to_netcdf(f'F:/LHSH/XLUcode/TCH_analysis_without GLEAM/TCH_result_{hh[kk]}_0124.nc')
print("文件保存成功: TCH_result_0124.nc")

# plot setting
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 11
zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
crs = ccrs.PlateCarree()
fig, axes = pplt.subplots(ncols=3, nrows=2, proj=crs,
                          sharex=False, sharey=False,
                          refaspect=1.5,
                          hspace=0.5)
axes.format(abc='(a)',
            grid=False,
            tickminor=False,
            titleloc='l')

levelsZ=np.linspace(0, 100, 11)
cmappZ = plt.get_cmap("rainbow")
normZ = mcolors.BoundaryNorm(boundaries=levelsZ, ncolors=cmappZ.N)
mapple = cm.ScalarMappable(norm=normZ, cmap=cmappZ)


result_ds0=result_ds['relative_uncertainty']
for k in range(5):
        data = result_ds0[:,:,k]
        data = data .rio.write_crs("EPSG:4326")
        shapefile = gpd.read_file('I://ERA5//boundary/TPBoundary_new(2021).shp')
        differ_clipped = data.rio.clip(shapefile.geometry, shapefile.crs)
        ax = axes[k]

        XX, YY = np.meshgrid(differ_clipped.x, differ_clipped.y)
        map = ax.contourf(XX, YY, differ_clipped.squeeze(), rasterized=True, levels=levelsZ,
                          cmap=plt.get_cmap("rainbow"), zorder=1, transform=ccrs.PlateCarree(), norm=normZ,
                          extend='both')

        province1 = shpreader.Reader('I://ERA5//boundary/TPBoundary_new(2021).shp').geometries()
        ax.add_geometries(province1, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1.0, zorder=10)
        scalebar = ScaleBar(5, units='km',
                            location='lower left')  # 1 is the length of the scale bar in the units specified (km)

        ax.add_artist(scalebar)
        ax.set_xticks([75, 85, 95, 105], crs=crs)
        ax.set_yticks([25, 30, 35, 40], crs=crs)
        ax.set_extent([68, 106, 25, 42], crs=crs)
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(length=2, width=0.3)
        ax.gridlines(draw_labels=False, zorder=2, linestyle='--', alpha=0)
        ax.set_title(f"{data.dataset.values} ", loc='left')
cax = plt.axes([0.05, 0.09, 0.7, 0.015])
cbar = plt.colorbar(mapple, cax=cax, orientation='horizontal', ticks=levelsZ)
cbar.set_label('Relative Uncertainty \n (W/M$^2$)')
cbar.ax.xaxis.set_label_coords(1.18, 0.85)
plt.tight_layout()
plt.savefig(f'F:/LHSH/XLUcode/TCH_analysis_without GLEAM/figure4_TCH_{hh[kk]}_0124.png', dpi=300, bbox_inches='tight')
plt.show()


print('finish')

