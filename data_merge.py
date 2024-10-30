import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

def merge_cyclone_sst(cyclone_path, sst_path, output_file_path):
    # Load cyclone data
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

    # change latitude to negative (southern hemisphere coordinates)
    data['lat_tenth'] = -data['lat_tenth']

    # load netCDF file
    nc_data = xr.open_dataset(sst_path)

    sst_values = []
    sst_values_before = {f'SST_before_{i}': [] for i in range(1, 16)}
    sst_values_after = {f'SST_after_{i}': [] for i in range(1, 16)}
    avg_sst_values = []
    rsst_values = []
    months = []
    seasons = []
    saffir_simpson_categories = []

    # Function to extract season from month
    def extract_season(month):
        if month in [12, 1, 2]:
            return 'DJF'
        elif month in [3, 4, 5]:
            return 'MAM'
        elif month in [6, 7, 8]:
            return 'JJA'
        else:
            return 'SON'

    # Function to determine Saffir-Simpson category
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

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing cyclone data"):
        # Extract the time, latitude, and longitude
        time_str = str(row['Time'])
        year = int(time_str[:4])
        month = int(time_str[4:6])
        day = int(time_str[6:8])
        hour = int(time_str[8:])
        time = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        lat = row['lat_tenth']
        lon = row['lon_tenth']

        # Extract month and season information
        months.append(month)
        seasons.append(extract_season(month))

        # Determine Saffir-Simpson category based on speed
        speed = row['Speed(knots)']
        saffir_simpson_categories.append(categorize_saffir_simpson(speed))

        # Select nearest time, latitude, and longitude for current SST value using interpolation
        try:
            sst_value = nc_data.sst.interp(time=time, lat=lat, lon=lon, method='nearest').values
            sst_values.append(sst_value.item() if sst_value.size > 0 else float('nan'))
        except Exception as e:
            print(f"Error finding SST for index {index}: {e}")
            sst_values.append(float('nan'))

        # SST values for the previous 15 days (using the same time of day)
        for i in range(1, 16):
            prev_time = time - pd.Timedelta(days=i)
            try:
                sst_value_before = nc_data.sst.interp(time=prev_time, lat=lat, lon=lon, method='nearest').values
                sst_values_before[f'SST_before_{i}'].append(sst_value_before.item() if sst_value_before.size > 0 else float('nan'))
            except Exception as e:
                print(f"Error finding SST_before_{i} for index {index}: {e}")
                sst_values_before[f'SST_before_{i}'].append(float('nan'))

        # SST values for the next 15 days (using the same time of day)
        for i in range(1, 16):
            next_time = time + pd.Timedelta(days=i)
            try:
                sst_value_after = nc_data.sst.interp(time=next_time, lat=lat, lon=lon, method='nearest').values
                sst_values_after[f'SST_after_{i}'].append(sst_value_after.item() if sst_value_after.size > 0 else float('nan'))
            except Exception as e:
                print(f"Error finding SST_after_{i} for index {index}: {e}")
                sst_values_after[f'SST_after_{i}'].append(float('nan'))

        # Calculate AVG_SST (average SST for the nearest hour across the entire month)
        try:
            # Find the nearest hour and location
            nearest_hour = nc_data.time.sel(time=time, method='nearest').dt.hour.values.item()
            month_start = time.replace(day=1, hour=0, minute=0, second=0)
            month_end = (month_start + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)
            avg_sst = nc_data.sst.sel(lat=lat, lon=lon, method='nearest').sel(time=slice(month_start, month_end)).where(nc_data['time.hour'] == nearest_hour).mean(dim='time', skipna=True).values
            avg_sst_values.append(avg_sst.item() if avg_sst.size > 0 else float('nan'))
        except Exception as e:
            print(f"Error calculating AVG_SST for index {index}: {e}")
            avg_sst_values.append(float('nan'))

        # Calculate RSST (current SST - AVG_SST)
        try:
            rsst = sst_values[-1] - avg_sst_values[-1] if not np.isnan(sst_values[-1]) and not np.isnan(avg_sst_values[-1]) else float('nan')
            rsst_values.append(rsst)
        except Exception as e:
            print(f"Error calculating RSST for index {index}: {e}")
            rsst_values.append(float('nan'))

    # Add SST, AVG_SST, RSST, month, season, Saffir-Simpson category, and SST_before/after columns to the dataframe
    data['SST'] = sst_values
    data['AVG_SST'] = avg_sst_values
    data['RSST'] = rsst_values
    data['Month'] = months
    data['Season'] = seasons
    data['Saffir_Simpson_Category'] = saffir_simpson_categories
    for key, values in sst_values_before.items():
        data[key] = values
    for key, values in sst_values_after.items():
        data[key] = values

    # save file
    try:
        data.to_csv(output_file_path, index=False)
    except PermissionError as e:
        print(f"PermissionError: {e}. Please check if the file is open or in use.")


# use this function to generate a csv file (example for SI and SP)
# merge_cyclone_sst('D:/pythonfile/cyclone/South_indian_hurricane.csv', 'D:/data/netcdf_files/SI_SST_data.nc', 'D:/pythonfile/cyclone/South_indian_hurricane_SST.csv')
# merge_cyclone_sst('D:/pythonfile/cyclone/South_pacific_hurricane.csv', 'D:/data/netcdf_files/SI_SST_data.nc', 'D:/pythonfile/cyclone/South_pacific_hurricane_SST.csv')