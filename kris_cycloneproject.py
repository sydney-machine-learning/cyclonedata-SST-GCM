import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io import shapereader
import seaborn as sns

def merge_cyclone_sst(cyclone_path, sst_path,output_file_path):
    file_path = cyclone_path
    data = pd.read_csv(file_path)

    # labeling cyclone
    data['Cycl_Seq'] = None
    current_number = data.iloc[0]['No. of Cycl']
    sequence_number = 1
    data.at[0, 'Cycl_Seq'] = sequence_number
    for i in range(1, len(data)):
        if data.at[i, 'No. of Cycl'] != current_number:
            sequence_number += 1
            current_number = data.at[i, 'No. of Cycl']
        data.at[i, 'Cycl_Seq'] = sequence_number

    data['No. of Cycl'] = data['Cycl_Seq']
    data.rename(columns={'No. of Cycl': 'No_cyclone'}, inplace=True)
    data.drop(columns=['Cycl_Seq'], inplace=True)

    # change latitude to negetive (southern hemisphere coordinates)
    data['lat_tenth'] = -data['lat_tenth']

    # new time format
    data['Year'] = data['Time'].astype(str).str[:4]
    data['Month'] = data['Time'].astype(str).str[4:6]
    data['Hour'] = data['Time'].astype(str).str[8:]
    data['New_Time'] = pd.to_datetime(data['Year'] + '-' + data['Month'] + '-01 ' + data['Hour'] + ':00:00')
    data['Time'] = data['Time'].astype(str)

    def extract_season(time_str):
        month = int(time_str[4:6])
        if month in [12, 1, 2]:
            season = 'DJF'
        elif month in [3, 4, 5]:
            season = 'MAM'
        elif month in [6, 7, 8]:
            season = 'JJA'
        else:
            season = 'SON'
        return season

    data['Season'] = data['Time'].map(extract_season)

    # create saffir simpson category column
    def categorize_saffir_simpson(speed):
        if speed < 82:
            return 1
        elif 82 <= speed < 95:
            return 2
        elif 95 <= speed < 112:
            return 3
        elif 112 <= speed < 136:
            return 4
        else:
            return 5

    data['Saffir-Simpson_Category'] = data['Speed(knots)'].apply(categorize_saffir_simpson)

    # load netCDF file
    nc_file_path = sst_path
    nc_data = xr.open_dataset(nc_file_path)
    data['New_Time'] = pd.to_datetime(data['New_Time'])
    sst_values = []
    for index, row in data.iterrows():
        # extract the time, latitude, and longitude
        time = row['New_Time']
        lat = row['lat_tenth']
        lon = row['lon_tenth']

        # nc info
        time_index = nc_data.time.sel(time=time, method='nearest').time
        lat_index = nc_data.latitude.sel(latitude=lat, method='nearest').latitude
        lon_index = nc_data.longitude.sel(longitude=lon, method='nearest').longitude

        # match
        sst_value = nc_data.sst.sel(time=time_index, latitude=lat_index, longitude=lon_index, method='nearest').values
        sst_values.append(sst_value[0] if sst_value.size > 0 else float('nan'))

    data['SST'] = sst_values

    # save file
    data.to_csv(output_file_path, index=False)

# use this function to generate a new csv file
# merge_cyclone_sst('D:/pythonfile/cyclone/South_indian_hurricane.csv','D:/pythonfile/cyclone/SI_SST.nc','D:/pythonfile/cyclone/Updated_South_indian_hurricane_with_SST.csv')

# ---------------------------------------------------------------------------------------------------------------------

def plot_cyclone_tracks_cartopy(path):
    data = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    etopo = shapereader.natural_earth(resolution='10m', category='physical', name='etopo1')
    etopo_data = xr.open_rasterio(etopo)
    etopo_plot = ax.imshow(etopo_data[0], origin='upper', transform=ccrs.PlateCarree(),
                           extent=[etopo_data.x.min(), etopo_data.x.max(), etopo_data.y.min(), etopo_data.y.max()],
                           cmap='terrain', alpha=0.5)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    ax.add_feature(cfeature.LAND, facecolor='coral')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES, facecolor='aqua')
    ax.set_extent([30, 130, -50, 20], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
                      xlocs=range(30, 130, 5), ylocs=range(-50, 20, 5))
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    for cyclone_id in data['No. of Cycl'].unique():
        track = data[data['No. of Cycl'] == cyclone_id]
        ax.plot(track['lon_tenth'], track['lat_tenth'], linewidth=0.5, color='red', transform=ccrs.Geodetic())

    ax.text(80, -5, "Indian Ocean", transform=ccrs.PlateCarree(), fontsize=12, verticalalignment='center',
            horizontalalignment='right')

    plt.colorbar(etopo_plot, ax=ax, orientation='horizontal', label='Elevation (meters)')
    plt.title("Cyclone Tracks in the South Indian Basin")
    plt.savefig('D:/pythonfile/cyclone/figure/cyclone_tracks_map.png', format='png', dpi=300)

# use this function to generate a new csv file
# plot_cyclone_tracks_cartopy('D:/pythonfile/cyclone/Updated_South_indian_hurricane_with_SST.csv')

# ---------------------------------------------------------------------------------------------------------------------

def plot_sst_SI(path):
    # calculate the SST over all region in South India Ocean
    ds = xr.open_dataset(path)
    mean_sst = ds['sst'].mean(dim=['time', 'expver'])
    mean_sst = mean_sst - 273.15

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    ax.add_feature(cfeature.LAND, facecolor='coral')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES, facecolor='aqua')
    ax.set_extent([30, 130, -50, 20], crs=ccrs.PlateCarree())

    sst_plot = ax.pcolormesh(ds['longitude'], ds['latitude'], mean_sst,
                             shading='auto', transform=ccrs.PlateCarree())

    plt.colorbar(sst_plot, ax=ax, orientation='vertical')

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
                      xlocs=range(30, 130, 5), ylocs=range(-50, 20, 5))
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.title('Average Sea Surface Temperature')
    plt.savefig('D:/pythonfile/cyclone/figure/SI_temp.png', format='png', dpi=300)

# use this function to generate a new csv file
# plot_sst_SI('D:/pythonfile/cyclone/SI_SST.nc')

# ---------------------------------------------------------------------------------------------------------------------

# def csvdata_region_month_sst(path_in,path_out,max_long,low_long,max_lat,low_lat):
#     ds = xr.open_dataset(path_in)
#     times = pd.to_datetime(ds['time'].values)
#     data = times[times == times.normalize()]
#     dataset = ds.sel(time=data)
#
#     filtered_data = dataset.where(
#         (dataset['longitude'] >= low_long) & (dataset['longitude'] <= max_long) &
#         (dataset['latitude'] >= low_lat) & (dataset['latitude'] <= max_lat), drop=True
#     )
#     filtered_data_df = filtered_data.to_dataframe().reset_index()
#     filtered_data_df['time'] = pd.to_datetime(filtered_data_df['time'])
#
#     monthly_avg_sst = filtered_data_df.groupby(filtered_data_df['time'].dt.to_period('M')).mean()['sst']
#     monthly_avg_sst_df = monthly_avg_sst.reset_index()
#     monthly_avg_sst_df.columns = ['Date', 'Average_SST']
#     monthly_avg_sst_df.to_csv(path_out, index=False)
#
#     data = pd.read_csv(path_out)
#     data['Date'] = pd.to_datetime(data['Date'])
#     data['Month_only'] = data['Date'].dt.month
#     monthly_avg_sst = data.groupby('Month_only')['Average_SST'].mean().reset_index()
#     monthly_avg_sst.columns = ['Month', 'Average_SST']
#     monthly_avg_sst['Average_SST'] = monthly_avg_sst['Average_SST'] - 273.15
#     monthly_avg_sst.to_csv(path_out, index=False)

def plot_month_sst_3r(path_csv,path_fig,path1,path2,path3):

    def csvdata_region_month_sst(path_in, path_out, max_long, low_long, max_lat, low_lat):
        ds = xr.open_dataset(path_in)
        times = pd.to_datetime(ds['time'].values)
        data = times[times == times.normalize()]
        dataset = ds.sel(time=data)

        filtered_data = dataset.where(
            (dataset['longitude'] >= low_long) & (dataset['longitude'] <= max_long) &
            (dataset['latitude'] >= low_lat) & (dataset['latitude'] <= max_lat), drop=True
        )
        filtered_data_df = filtered_data.to_dataframe().reset_index()
        filtered_data_df['time'] = pd.to_datetime(filtered_data_df['time'])

        monthly_avg_sst = filtered_data_df.groupby(filtered_data_df['time'].dt.to_period('M')).mean()['sst']
        monthly_avg_sst_df = monthly_avg_sst.reset_index()
        monthly_avg_sst_df.columns = ['Date', 'Average_SST']
        monthly_avg_sst_df.to_csv(path_out, index=False)

        data = pd.read_csv(path_out)
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month_only'] = data['Date'].dt.month
        monthly_avg_sst = data.groupby('Month_only')['Average_SST'].mean().reset_index()
        monthly_avg_sst.columns = ['Month', 'Average_SST']
        monthly_avg_sst['Average_SST'] = monthly_avg_sst['Average_SST'] - 273.15
        monthly_avg_sst.to_csv(path_out, index=False)

    csvdata_region_month_sst('D:/pythonfile/cyclone/SI_SST.nc', path1, 120, 115, -15,
                             -20)
    csvdata_region_month_sst('D:/pythonfile/cyclone/SI_SST.nc', path2, 65, 55, -10,
                             -20)
    csvdata_region_month_sst('D:/pythonfile/cyclone/SI_SST.nc', path3, 105, 95, -10,
                             -20)

    data1 = pd.read_csv(path1,usecols=['Month','Average_SST'])
    data2 = pd.read_csv(path2,usecols=['Average_SST'])
    data3 = pd.read_csv(path3, usecols=['Average_SST'])

    data1 = data1.rename(columns={'Average_SST': 'average_sst_r1'})
    data2 = data2.rename(columns={'Average_SST': 'average_sst_r2'})
    data3 = data3.rename(columns={'Average_SST': 'average_sst_r3'})

    combined_data = pd.concat([data1,data2,data3], axis=1)
    combined_data.to_csv(path_csv, index=False)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='average_sst_r1', data=combined_data, marker='o', label='R1_Average_SST')
    sns.lineplot(x='Month', y='average_sst_r2', data=combined_data, marker='o', label='R2_Average_SST')
    sns.lineplot(x='Month', y='average_sst_r3', data=combined_data, marker='o', label='R3_Average_SST')

    plt.xticks(ticks=range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.title('Monthly Average SST')
    plt.xlabel('Month')
    plt.ylabel('Average SST')
    plt.legend(title='Regions')
    plt.savefig(path_fig, format='png')

# use this function to generate new csv files and figure
# plot_month_sst_3r('D:/pythonfile/cyclone/monthsst_allr.csv','D:/pythonfile/cyclone/figure/SI_monthsst_3region.png','D:/pythonfile/cyclone/monthsst_r1.csv','D:/pythonfile/cyclone/monthsst_r2.csv','D:/pythonfile/cyclone/monthsst_r3.csv')


# ---------------------------------------------------------------------------------------------------------------------

#%%

import pandas as pd
import matplotlib.pyplot as plt

def abc(path_in, fig_out):
    data = pd.read_csv(path_in)

    data_filtered = data[['Month', 'Saffir-Simpson_Category']]
    data_grouped = data_filtered.groupby(['Month', 'Saffir-Simpson_Category']).size().unstack(fill_value=0)
    colors = ['#d4e157', '#a0d568', '#6abf69', '#388e3c', '#1b5e20']
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.8  # 设置柱状宽度
    bottom = None

    for category, color in zip(data_grouped.columns, colors):
        if bottom is None:
            bottom = data_grouped[category]
            ax.bar(data_grouped.index, data_grouped[category], label=category, color=color, width=bar_width)
        else:
            ax.bar(data_grouped.index, data_grouped[category], label=category, color=color, bottom=bottom, width=bar_width)
            bottom += data_grouped[category]

    plt.title('Monthly Distribution of TC Categories')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Number of TC Occurrences', fontsize=14)
    plt.xticks(ticks=range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=18)
    plt.yticks(fontsize=20)
    plt.legend(title='Category', title_fontsize='15', fontsize='18')

    plt.savefig(fig_out, format='png', dpi=300)

# use this function to generate polt
# abc('D:/pythonfile/cyclone/Updated_South_indian_hurricane_with_SST.csv', 'D:/pythonfile/cyclone/figure/SI_month_ssc.png')


