# ----------------------------------------------------------------------------------------------------------------------
# Download all nc files from NOAA
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import xarray as xr

# access NOAA
BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
# download
DOWNLOAD_DIR = "netcdf_files"

# Create a directory to store downloaded files
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def download_netcdf_files(base_url, download_dir, start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime("%Y%m")
        month_url = urljoin(base_url, current_date.strftime("%Y%m/"))
        month_download_dir = os.path.join(download_dir, year_month)

        if not os.path.exists(month_download_dir):
            os.makedirs(month_download_dir)

        try:
            response = requests.get(month_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                if href and href.endswith('.nc'):
                    file_url = urljoin(month_url, href)
                    file_name = os.path.join(month_download_dir, os.path.basename(href))
                    download_file(file_url, file_name)

        except requests.RequestException as e:
            print(f"error: {e}")

        # next month
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)

def download_file(url, file_name):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded: {file_name}")
    except requests.RequestException as e:
        print(f"error: {e}")

if __name__ == "__main__":
    start_date = datetime(2014, 10, 1)
    end_date = datetime(2024, 9, 1)
    download_netcdf_files(BASE_URL, DOWNLOAD_DIR, start_date, end_date)

# ----------------------------------------------------------------------------------------------------------------------
# merge monthly data

# path of input
BASE_FOLDER_PATH = "D:/data/netcdf_files"
# path of output
OUTPUT_FOLDER = "D:/data/netcdf_files/month_SST"

# 创建保存合并文件的目录
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def merge_netcdf_files(folder_path, output_file):
    # find nc file
    netcdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]

    if not netcdf_files:
        print(f"{folder_path} cant find netCDF ")
        return

    try:
        # load and merge
        datasets = [xr.open_dataset(f) for f in netcdf_files]
        merged_dataset = xr.concat(datasets, dim='time')

        # save
        merged_dataset.to_netcdf(output_file)
        print(f"Successfully merged and save to : {output_file}")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    # Traverse all folders
    for year in range(1981, 2025):
        for month in range(1, 13):
            # skip 1981 JAN to AUG
            if year == 1981 and month < 9:
                continue
            # skip 2014 OCT to DEC
            if year == 2024 and month > 9:
                break

            # name
            folder_name = f"{year}{month:02d}"
            folder_path = os.path.join(BASE_FOLDER_PATH, folder_name)
            output_file = os.path.join(OUTPUT_FOLDER, f"{folder_name}_merged.nc")

            # merge monthly nc file
            if os.path.exists(folder_path):
                merge_netcdf_files(folder_path, output_file)

# ----------------------------------------------------------------------------------------------------------------------
# merge daily data

# path of input
MERGED_FOLDER_PATH = "D:/data/netcdf_files/month_SST"
# path of output
FINAL_OUTPUT_FILE = "D:/data/netcdf_files/SI_SST_data.nc"


def merge_all_monthly_files(merged_folder_path, output_file):

    netcdf_files = [os.path.join(merged_folder_path, f) for f in os.listdir(merged_folder_path) if
                    f.endswith('_merged.nc')]

    if not netcdf_files:
        print("cant find nc file。")
        return

    try:
        merged_dataset = xr.open_mfdataset(netcdf_files, concat_dim='time', combine='nested')
        merged_dataset.to_netcdf(output_file)
        print(f"Successfully downloaded: {output_file}")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    merge_all_monthly_files(MERGED_FOLDER_PATH, FINAL_OUTPUT_FILE)