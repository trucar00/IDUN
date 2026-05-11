import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_mix_2023.parquet")
#df = df.loc[df["trajectory_id"] == "257088050-9"].copy()
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

def plot_mmsi(mmsi):
    new_dd = df[df["mmsi"] == mmsi]
    new_dd = new_dd.sort_values(by="date_time_utc")

    fig, ax = plt.subplots(figsize=(14, 7))

   
    # NEW
    for traj_id, d in new_dd.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc")
        ax.plot(d["lon"], d["lat"], ".", markersize=2)

    ax.set_title("New cleaning")

    # Equal aspect ratio (important for geo plots)
    plt.suptitle(f"MMSI: {mmsi}")
    plt.tight_layout()
    plt.show()


# Example: plot one MMSI

def haversine(lat1, lon1, lat2, lon2, dt):
    R = 6371000 # Radius of the earth in meters

    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) + 
         pow(np.sin(dLon / 2), 2) * 
             np.cos(lat1) * np.cos(lat2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    speed = (dist/dt)

    return dist, speed

def remove_spikes_three_point(df, ratio_threshold=0.5, min_perp=5):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"])

    g = df.groupby("trajectory_id", sort=False)
    lat_prev = g["lat"].shift(1);  lon_prev = g["lon"].shift(1)
    lat_next = g["lat"].shift(-1); lon_next = g["lon"].shift(-1)

    d_ab, _ = haversine(lat_prev, lon_prev, df["lat"], df["lon"], 1)
    d_bc, _ = haversine(df["lat"], df["lon"], lat_next, lon_next, 1)
    d_ac, _ = haversine(lat_prev, lon_prev, lat_next, lon_next, 1)

    # Heron -> perpendicular distance from B to line AC
    s = 0.5 * (d_ab + d_bc + d_ac)
    area = np.sqrt(np.clip(s * (s - d_ab) * (s - d_bc) * (s - d_ac), 0, None))
    d_ac_safe = d_ac.replace(0, np.nan)
    perp_dist = (2 * area) / d_ac_safe
    ratio = perp_dist / d_ac_safe
    print(perp_dist, d_ac, ratio)

    spike = ((ratio > ratio_threshold) & (perp_dist > min_perp)).fillna(False)

    print(f"Removed {spike.sum():,} three-point spikes")
    return df.loc[~spike].drop(columns=["lat_prev", "lon_prev",
                                        "lat_next", "lon_next"],
                               errors="ignore")

print(haversine(61.4800, 1.80169, 61.48145, 1.80270, 1))
#print(haversine(61.482042, 1.80185, 61.482271, 1.801519, 1))



for mmsi, d in df.groupby("mmsi"):
    """ for t, dd in d.groupby("trajectory_id"):
        print(dd)
        #dd_clean = remove_spikes_three_point(dd.iloc[9328:9340]) # [9180:9200] # [9328:9340]
        plt.scatter(dd["lon"], dd["lat"], s=10)
        #plt.scatter(dd_clean["lon"], dd_clean["lat"], s=2, color="red")
        plt.title(f"{t}")
        plt.show() """
    plot_mmsi(mmsi)