import os
import requests
import py7zr
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ------------------------------------------------------------
# General download function: if save_path does not exist, download from URL and save
# ------------------------------------------------------------
def download_from_zenodo(url: str, save_path: str):
    """
    If save_path does not exist locally, download from url and save to save_path.
    url should be in the form:
      https://zenodo.org/records/15485947/files/South_indian_hurricane.csv?download=1
    save_path is recommended to be under the project's data/ directory
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading:\n  {url}\n  -> {save_path}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("  Download complete.")
    else:
        print(f"File already exists: {save_path}")


# ------------------------------------------------------------
# Download and merge split .7z files, then extract to NetCDF, return .nc path
# ------------------------------------------------------------
def fetch_and_extract_sst_7z(parts_urls, download_dir, extract_dir):
    """
    parts_urls: List of 3 Zenodo URLs, e.g.:
      [
        "https://zenodo.org/records/15485947/files/SST_data.7z.001?download=1",
        "https://zenodo.org/records/15485947/files/SST_data.7z.002?download=1",
        "https://zenodo.org/records/15485947/files/SST_data.7z.003?download=1"
      ]
    download_dir: Local directory to save these .7z parts
    extract_dir: Directory to extract the resulting .nc files

    Returns: Full path to the first .nc file found after extraction
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    local_parts = []
    # 1. Download each part
    for url in parts_urls:
        filename = url.split("/")[-1].split("?")[0]  # e.g. SST_data.7z.001
        local_path = os.path.join(download_dir, filename)
        download_from_zenodo(url, local_path)
        local_parts.append(local_path)

    # 2. Sort and take the first part so py7zr can recognize multi-volume archive
    first_part = sorted(local_parts)[0]

    # 3. Extract with py7zr
    print(f"\nExtracting 7z archive from {first_part} into {extract_dir} ...")
    with py7zr.SevenZipFile(first_part, mode='r') as archive:
        archive.extractall(path=extract_dir)
    print("Extraction complete.\n")

    # 4. Find and return the first .nc file in extract_dir
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith(".nc"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"No .nc file found in {extract_dir}.")


# ------------------------------------------------------------
# Original merge_cyclone_sst function
# ------------------------------------------------------------
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
            avg_sst = nc_data.sst.sel(lat=lat, lon=lon, method='nearest') \
                .sel(time=slice(month_start, month_end)) \
                .where(nc_data['time.hour'] == nearest_hour) \
                .mean(dim='time', skipna=True).values
            avg_sst_values.append(avg_sst.item() if avg_sst.size > 0 else float('nan'))
        except Exception as e:
            print(f"Error calculating AVG_SST for index {index}: {e}")
            avg_sst_values.append(float('nan'))

        # Calculate RSST (current SST - AVG_SST)
        try:
            rsst = sst_values[-1] - avg_sst_values[-1] \
                if not pd.isna(sst_values[-1]) and not pd.isna(avg_sst_values[-1]) else float('nan')
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


# ------------------------------------------------------------
# main(): Download SST .7z parts, extract to .nc, download cyclone CSVs, then merge
# ------------------------------------------------------------
def main():
    # -- (A) Download and merge SST_data.7z parts, then extract .nc -- #
    sst_parts = [
        "https://zenodo.org/records/15485947/files/SST_data.7z.001?download=1",
        "https://zenodo.org/records/15485947/files/SST_data.7z.002?download=1",
        "https://zenodo.org/records/15485947/files/SST_data.7z.003?download=1"
    ]
    sst_download_dir = "data/sst_parts"
    sst_extract_dir = "data/sst_extracted"
    sst_nc_path = fetch_and_extract_sst_7z(sst_parts, sst_download_dir, sst_extract_dir)
    print(f"Extracted NetCDF file: {sst_nc_path}\n")

    # -- (B) Download raw cyclone CSVs for each basin -- #
    zenodo_csv_urls = {
        "data/cyclone/South_indian_hurricane.csv":
            "https://zenodo.org/records/15485947/files/South_indian_hurricane.csv?download=1",
        "data/cyclone/South_pacific_hurricane.csv":
            "https://zenodo.org/records/15485947/files/South_pacific_hurricane.csv?download=1",
        "data/cyclone/North_westpacificocean_hurricane.csv":
            "https://zenodo.org/records/15485947/files/North_westpacificocean_hurricane.csv?download=1",
        "data/cyclone/North_Indian_hurricane.csv":
            "https://zenodo.org/records/15485947/files/North_Indian_hurricane.csv?download=1"
    }
    for local_path, url in zenodo_csv_urls.items():
        download_from_zenodo(url, local_path)
    print()

    # -- (C) Ensure output directory for merged CSVs exists -- #
    os.makedirs("data/cyclone_merged", exist_ok=True)

    # -- (D) Call merge_cyclone_sst to generate SST-merged CSVs -- #
    merge_cyclone_sst(
        cyclone_path="data/cyclone/South_indian_hurricane.csv",
        sst_path=sst_nc_path,
        output_file_path="data/cyclone_merged/South_indian_hurricane_SST.csv"
    )

    merge_cyclone_sst(
        cyclone_path="data/cyclone/South_pacific_hurricane.csv",
        sst_path=sst_nc_path,
        output_file_path="data/cyclone_merged/South_pacific_hurricane_SST.csv"
    )

    merge_cyclone_sst(
        cyclone_path="data/cyclone/North_westpacificocean_hurricane.csv",
        sst_path=sst_nc_path,
        output_file_path="data/cyclone_merged/North_westpacificocean_hurricane_SST.csv"
    )

    merge_cyclone_sst(
        cyclone_path="data/cyclone/North_Indian_hurricane.csv",
        sst_path=sst_nc_path,
        output_file_path="data/cyclone_merged/North_Indian_hurricane_SST.csv"
    )

    print("\nAll merge tasks completed. The merged CSVs are under data/cyclone_merged/")

if __name__ == "__main__":
    main()
