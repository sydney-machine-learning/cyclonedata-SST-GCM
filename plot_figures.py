import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from pygam import LinearGAM, s
import matplotlib.patches as mpatches
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# PLOT figure 1
# ------------------------------------------------------------
def plot_figure1(path):
    ds = xr.open_dataset(path)
    mean_sst = ds['sst'].mean(dim=['time', 'expver'])
    mean_sst = mean_sst - 273.15

    fig, ax = plt.subplots(figsize=(12, 10))

    # create map
    m = Basemap(projection='merc',
                llcrnrlat=-50, urcrnrlat=0,
                llcrnrlon=30, urcrnrlon=130,
                resolution='i', ax=ax)

    # add features
    m.shadedrelief()
    m.drawcoastlines()
    m.drawcountries()
    m.drawrivers(color='blue')

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # grid
    parallels = np.arange(-50., 0., 10.)
    meridians = np.arange(30., 130., 10.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=14)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=14)

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = m(lon2d, lat2d)
    sst_plot = m.pcolormesh(x, y, mean_sst, shading='auto', cmap='coolwarm')
    cbar = plt.colorbar(sst_plot, ax=ax, orientation='vertical', pad=0.05, shrink=0.4)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Sea Surface Temperature (°C)', fontsize=14)

    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure1.png', format='png', dpi=300)
    plt.show()


# ------------------------------------------------------------
# PLOT figure 2
# ------------------------------------------------------------
def plot_figure2(path):
    ds = xr.open_dataset(path)
    regions = {
        "South Indian Ocean": {"lat": (-40, 0), "lon": (20, 90)},
        "Northwest Pacific Ocean": {"lat": (0, 30), "lon": (100, 180)},
        "South Pacific Ocean": {"lat": (-40, 0), "lon": (160, 240)},
        "North Indian Ocean": {"lat": (0, 30), "lon": (45, 100)}
    }
    ticks = np.arange(1982, 2025, 6)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=False)
    subplot_info = [
        ("a", "South Indian Ocean Basin"),
        ("b", "Northwest Pacific Ocean Basin"),
        ("c", "South Pacific Ocean Basin"),
        ("d", "North Indian Ocean Basin")
    ]

    for i, ((region_name, bounds), (label, subtitle)) in enumerate(zip(regions.items(), subplot_info)):
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]
        region_ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        monthly_data = (
            region_ds.sst
            .mean(dim=["lat", "lon", "zlev"])
            .sel(time=slice("1982", "2024"))
        )
        annual_mean = monthly_data.groupby("time.year").mean()
        annual_std = monthly_data.groupby("time.year").std()
        count = monthly_data.groupby("time.year").count()
        se = annual_std / np.sqrt(count)
        ci_lower = annual_mean - 1.96 * se
        ci_upper = annual_mean + 1.96 * se

        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.plot(annual_mean.year, annual_mean, marker='o', color='blue', label='Mean')
        ax.fill_between(annual_mean.year, ci_lower, ci_upper, color='gray', alpha=0.3, label='95% CI')
        ax.grid(True)
        ax.set_ylabel("SST (°C)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45)
        ax.set_xlabel("Year")
        ax.text(
            0.5, -0.25,
            f"({label}) {subtitle}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure2.png', dpi=300, format='png')
    plt.show()


# ------------------------------------------------------------
# PLOT figure 3
# ------------------------------------------------------------
def plot_figure3(path, sample_fraction=0.3):
    data = pd.read_csv(path)
    cmap = ListedColormap([
        'green',   # Category 1
        'yellow',  # Category 2
        'orange',  # Category 3
        'red',     # Category 4
        'darkred'  # Category 5
    ])
    fig, ax = plt.subplots(figsize=(12, 10))
    m = Basemap(projection='merc',
                llcrnrlat=-40, urcrnrlat=0,
                llcrnrlon=30, urcrnrlon=130,
                resolution='i',
                ax=ax)

    m.shadedrelief()
    m.drawcoastlines()
    m.drawcountries()
    m.drawrivers(color='blue')

    for spine in ax.spines.values():
        spine.set_linewidth(1)
    parallels = np.arange(-40., 0., 10.)
    meridians = np.arange(30., 130., 10.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=14)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=14)

    # --- NEW PART: Select a subset of cyclone tracks ---
    unique_cyclones = data['No_cyclone'].unique()
    # set random seed for reproducibility
    np.random.seed(42)
    # randomly sample a fraction of cyclone IDs
    sample_size = int(sample_fraction * len(unique_cyclones))
    selected_cyclones = np.random.choice(unique_cyclones, size=sample_size, replace=False)
    # ------------------------------------------------------

    # loop over selected cyclones only
    for cyclone_id in selected_cyclones:
        track = data[data['No_cyclone'] == cyclone_id]
        lon = track['lon_tenth'].values  # Longitude in degrees
        lat = track['lat_tenth'].values   # Latitude in degrees
        categories = track['Saffir_Simpson_Category'].values

        # project the latitude and longitude data to the Basemap coordinate system
        x, y = m(lon, lat)

        # plot the track segments with color corresponding to the category
        for i in range(len(x) - 1):
            color = cmap(categories[i] - 1)  # Categories range from 1 to 5
            m.plot(x[i:i+2], y[i:i+2], color=color, linewidth=1)

    # add a vertical colorbar on the right side and shrink its height
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax,
                        orientation='vertical', pad=0.05, shrink=0.4)
    cbar.set_label('Saffir-Simpson Category', rotation=90, labelpad=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([1, 2, 3, 4, 5])

    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure3.png', format='png', dpi=300)
    plt.show()


# ------------------------------------------------------------
# PLOT figure 4
# ------------------------------------------------------------
def plot_figure4():
    file_paths = [
        "data/cyclone_merged/South_indian_hurricane_SST.csv",
        "data/cyclone_merged/South_pacific_hurricane_SST.csv",
        "data/cyclone_merged/North_Indian_hurricane_SST.csv",
        "data/cyclone_merged/North_westpacificocean_hurricane_SST.csv"
    ]

    basins = ['South Indian', 'South Pacific', 'North Indian', 'Northwest Pacific']
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]

    def get_decade(year):
        if 1980 <= year < 1990:
            return '1980-1990'
        elif 1990 <= year < 2000:
            return '1990-2000'
        elif 2000 <= year < 2010:
            return '2000-2010'
        elif 2010 <= year < 2020:
            return '2010-2020'
        else:  # filter out years that don't match
            return None

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    fig.suptitle('Cyclone Category Distribution by Basin and Decade (1980-2020)', fontsize=16, y=1.02)
    colors = [plt.cm.Reds(i) for i in np.linspace(0.4, 0.8, 5)]

    for i, (df, basin) in enumerate(zip(dataframes, basins)):
        df['Year'] = pd.to_datetime(df['Time'], format='%Y%m%d%H').dt.year
        df['Decade'] = df['Year'].apply(get_decade)
        df = df.dropna(subset=['Decade'])
        df = df[(df['Speed(knots)'] != 999) & (df['Speed(knots)'] != -999)]
        max_categories = df.groupby('No_cyclone')['Saffir_Simpson_Category'].max()
        df_max = df[['No_cyclone', 'Decade']].drop_duplicates()
        df_max['Max_Category'] = df_max['No_cyclone'].map(max_categories)
        category_distribution = df_max.groupby(['Decade', 'Max_Category']).size().unstack(fill_value=0)
        all_decades = ['1980-1990', '1990-2000', '2000-2010', '2010-2020']
        category_distribution = category_distribution.reindex(all_decades, fill_value=0)
        ax = axes[i//2, i%2]
        category_distribution.plot(kind='bar',
                                   stacked=True,
                                   ax=ax,
                                   color=colors,
                                   legend=False)

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(subplot_labels[i], fontsize=14, pad=10)
        ax.set_xlabel('', fontsize=1)
        ax.set_ylabel('Number of Cyclones', fontsize=14)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for container in ax.containers:
            ax.bar_label(container, label_type='center', fmt='%d', color='black', fontsize=15)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels,
               title='Category',
               loc='lower right',
               bbox_to_anchor=(0.98, 0.29),
               framealpha=0.9)

    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()

    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure4.png', format='png', dpi=300)
    plt.show()


# ------------------------------------------------------------
# PLOT figure 6
# ------------------------------------------------------------
def plot_figure6():
    file_paths = [
        "data/cyclone_merged/South_indian_hurricane_SST.csv",
        "data/cyclone_merged/South_pacific_hurricane_SST.csv",
        "data/cyclone_merged/North_Indian_hurricane_SST.csv",
        "data/cyclone_merged/North_westpacificocean_hurricane_SST.csv"
    ]
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['South Indian', 'South Pacific', 'North Indian', 'North West Pacific']
    decades = ['1981-1990', '1991-2000', '2001-2010', '2011-2020']
    subplot_info = [
        ("a", "Proportions of tropical cyclones: 1981–1990"),
        ("b", "Proportions of tropical cyclones: 1991–2000"),
        ("c", "Proportions of tropical cyclones: 2001–2010"),
        ("d", "Proportions of tropical cyclones: 2011–2020"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, decade in enumerate(decades):
        ax = axes[i // 2, i % 2]
        for idx, file_path in enumerate(file_paths):
            df = pd.read_csv(file_path)
            if 'Time' not in df.columns:
                continue
            df['Year'] = df['Time'].astype(str).str[:4].astype(int)
            df = df.dropna(subset=['Year', 'Saffir_Simpson_Category', 'No_cyclone'])
            df = df[(df['Speed(knots)'] != 999) & (df['Speed(knots)'] != -999)]
            start, end = map(int, decade.split('-'))
            decade_data = df[(df['Year'] >= start) & (df['Year'] <= end)]
            max_categories = decade_data.groupby('No_cyclone')['Saffir_Simpson_Category'].max()
            if len(max_categories) == 0:
                continue
            proportions = [(max_categories >= k).sum() / len(max_categories) for k in range(1, 6)]
            x = np.arange(1, 6)
            f = interp1d(x, proportions, kind='cubic', fill_value="extrapolate")
            x_smooth = np.linspace(1, 5, 500)
            y_smooth = f(x_smooth)
            ax.plot(x_smooth, y_smooth, color=colors[idx], label=labels[idx], linewidth=2)
            ax.fill_between(x_smooth, y_smooth, color=colors[idx], alpha=0.2)

        ax.set_ylabel("Proportion")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1)
        label, subtitle = subplot_info[i]
        ax.text(
            0.5, -0.2,
            f"({label}) {subtitle}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

    handles = [plt.Line2D([], [], color=c, label=l, linewidth=2) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, title="Basin", loc='lower center', ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure6.png', dpi=300)
    plt.show()


# ------------------------------------------------------------
# PLOT figure 7
# ------------------------------------------------------------
def plot_figure7():
    file_paths = [
        "data/cyclone_merged/South_indian_hurricane_SST.csv",
        "data/cyclone_merged/South_pacific_hurricane_SST.csv",
        "data/cyclone_merged/North_Indian_hurricane_SST.csv",
        "data/cyclone_merged/North_westpacificocean_hurricane_SST.csv"
    ]

    colors = ['red', 'blue', 'green', 'purple']
    labels = ['SI', 'SP', 'NI', 'NW']
    time_points = np.arange(-15, 16)
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    subplot_info = [
        ("a", "Time period: DJF"),
        ("b", "Time period: MAM"),
        ("c", "Time period: JJA"),
        ("d", "Time period: SON")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, season in enumerate(season_order):
        ax = axes[idx // 2, idx % 2]
        for i, file_path in enumerate(file_paths):
            df = pd.read_csv(file_path)
            df = df.dropna(subset=[f'SST_before_{j}' for j in range(1, 16)] + [f'SST_after_{j}' for j in range(1, 16)])
            if season not in df['Season'].unique():
                continue
            season_data = df[df['Season'] == season]
            data = []
            for _, row in season_data.iterrows():
                for t in time_points:
                    if t < 0:
                        SST_value = row[f'SST_before_{abs(t)}']
                    elif t > 0:
                        SST_value = row[f'SST_after_{t}']
                    else:
                        SST_value = row['SST']
                    data.append([t, SST_value])

            data_df = pd.DataFrame(data, columns=['Time', 'SST'])
            gam = LinearGAM(s(0)).fit(data_df['Time'], data_df['SST'])
            forecast = gam.predict(time_points)
            ax.plot(time_points, forecast, label=labels[i], color=colors[i])

        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_xlabel("Days from Cyclone Event")
        ax.set_ylabel("SST")
        ax.set_xticks(range(-15, 16, 5))

        # Place the subplot label closer to the bottom edge
        label, subtitle = subplot_info[idx]
        ax.text(
            0.5, -0.15,                   # moved y from -0.2 up to -0.15
            f"({label}) {subtitle}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

    legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    axes[0, 1].legend(handles=legend_handles, loc='upper right', fontsize=10, title="Basin")

    plt.tight_layout()
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure7.png', dpi=300)
    plt.show()


# ------------------------------------------------------------
# PLOT figure 8
# ------------------------------------------------------------
def plot_figure8():
    file_paths = [
        "data/cyclone_merged/South_indian_hurricane_SST.csv",
        "data/cyclone_merged/South_pacific_hurricane_SST.csv",
        "data/cyclone_merged/North_Indian_hurricane_SST.csv",
        "data/cyclone_merged/North_westpacificocean_hurricane_SST.csv"
    ]
    basin_subtitles = [
        ("a", "South Indian Basin"),
        ("b", "South Pacific Basin"),
        ("c", "North Indian Basin"),
        ("d", "North West Pacific Basin")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_light = ['lightblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightpink']
    colors_dark = ['blue', 'green', 'red', 'cyan', 'magenta']

    for i, (file_path, (label, subtitle)) in enumerate(zip(file_paths, basin_subtitles)):
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['SST'])
        df = df[(df['Speed(knots)'] >= 0) & (df['Speed(knots)'] != 999)]
        ax = axes[i // 2, i % 2]

        for category, c_light, c_dark in zip(range(1, 6), colors_light, colors_dark):
            cat_data = df[df['Saffir_Simpson_Category'] == category]
            sst = cat_data['SST'].values.reshape(-1, 1)
            ws = cat_data['Speed(knots)'].values
            sst_range = np.linspace(sst.min(), sst.max(), 100).reshape(-1, 1)

            ax.scatter(sst, ws, color=c_light, alpha=0.5)

            svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            svr.fit(sst, ws)
            pred_svr = svr.predict(sst_range)
            ax.plot(sst_range, pred_svr, color=c_dark, linestyle='--')

            lin = LinearRegression()
            lin.fit(sst, ws)
            pred_lin = lin.predict(sst_range)
            ax.plot(sst_range, pred_lin, color=c_dark, linestyle='-')

        ax.set_xlabel('Sea Surface Temperature (SST)')
        ax.set_ylabel('Wind Speed (knots)')
        ax.grid(True)

        ax.text(
            0.5, -0.15,
            f"({label}) {subtitle}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

        if i == 0:
            handles = [
                plt.Line2D([], [], color=colors_dark[j], linestyle='-', label=f'Cat {j+1} (LR)')
                for j in range(5)
            ]
            handles += [
                plt.Line2D([], [], color=colors_dark[j], linestyle='--', label=f'Cat {j+1} (SVR)')
                for j in range(5)
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=8, title='Category')

    plt.tight_layout()
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/Figure8.png', dpi=300)
    plt.show()

    
# ------------------------------------------------------------
# Entry point: generate all Figures with one command
# ------------------------------------------------------------
if __name__ == '__main__':
    # Assume SST .nc file is located at data/sst_extracted/SST_data.nc
    nc_path = 'data/sst_extracted/SST_data.nc'
    # Assume merged South Indian CSV is located at data/cyclone_merged/South_indian_hurricane_SST.csv
    cyclone_csv_path = 'data/cyclone_merged/South_indian_hurricane_SST.csv'

    # Generate Figure1
    # plot_figure1(nc_path)

    # Generate Figure2
    # plot_figure2(nc_path)

    # Generate Figure3
    # plot_figure3(cyclone_csv_path, sample_fraction=0.3)

    # Generate Figure4
    # plot_figure4()

    # Generate Figure6
    # plot_figure6()

    # Generate Figure7
    # plot_figure7()

    # Generate Figure8
    # plot_figure8()