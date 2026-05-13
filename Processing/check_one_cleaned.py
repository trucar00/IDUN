import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def haversine(lat1, lon1, lat2, lon2, dt):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    speed = (dist/dt) #* 1.94384 # Convert m/s to knots

    return dist, speed

df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_2022_test.parquet")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = df.sort_values(by="date_time_utc")

def plot_trajectory_pairs(mmsi):
    d_mmsi = df[df["mmsi"] == mmsi].copy()
    d_mmsi["date_time_utc"] = pd.to_datetime(d_mmsi["date_time_utc"])
    d_mmsi = d_mmsi.sort_values("date_time_utc")

    # Get trajectories ordered by start time
    traj_summary = (
        d_mmsi.groupby("trajectory_id")
        .agg(
            start_time=("date_time_utc", "min"),
            end_time=("date_time_utc", "max"),
            start_lat=("lat", "first"),
            start_lon=("lon", "first"),
            end_lat=("lat", "last"),
            end_lon=("lon", "last"),
            n=("lat", "size"),
        )
        .sort_values("start_time")
        .reset_index()
    )

    for i in range(len(traj_summary) - 1):
        t1 = traj_summary.iloc[i]
        t2 = traj_summary.iloc[i + 1]

        gap_s = (t2["start_time"] - t1["end_time"]).total_seconds()

        dist_m, speed_ms = haversine(
            t1["end_lat"], t1["end_lon"],
            t2["start_lat"], t2["start_lon"],
            gap_s
        )

        speed_knots = speed_ms * 1.94384

        print("\n" + "-" * 60)
        print(f"MMSI: {mmsi}")
        print(f"Pair: {t1['trajectory_id']}  ->  {t2['trajectory_id']}")
        print(f"End time traj 1:   {t1['end_time']}")
        print(f"Start time traj 2: {t2['start_time']}")
        print(f"dt: {gap_s:.1f} s = {gap_s / 60:.2f} min")
        print(f"distance: {dist_m:.2f} m")
        print(f"implied speed: {speed_ms:.2f} m/s = {speed_knots:.2f} knots")

        fig, ax = plt.subplots(figsize=(10, 7))

        d1 = d_mmsi[d_mmsi["trajectory_id"] == t1["trajectory_id"]].sort_values("date_time_utc")
        d2 = d_mmsi[d_mmsi["trajectory_id"] == t2["trajectory_id"]].sort_values("date_time_utc")

        ax.plot(d1["lon"], d1["lat"], ".", markersize=3, label=t1["trajectory_id"])
        ax.plot(d2["lon"], d2["lat"], ".", markersize=3, label=t2["trajectory_id"])

        # Mark connection from end of traj 1 to start of traj 2
        ax.plot(
            [t1["end_lon"], t2["start_lon"]],
            [t1["end_lat"], t2["start_lat"]],
            "--",
            linewidth=1,
            label="gap"
        )

        ax.scatter(t1["end_lon"], t1["end_lat"], s=60, marker="x", label="end traj 1")
        ax.scatter(t2["start_lon"], t2["start_lat"], s=60, marker="o", facecolors="none", label="start traj 2")

        ax.set_title(
            f"{mmsi}: {t1['trajectory_id']} -> {t2['trajectory_id']}\n"
            f"dt={gap_s/60:.2f} min, speed={speed_ms:.2f} m/s ({speed_knots:.2f} kn)"
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.legend()
        plt.show()

plot_trajectory_pairs(257062150)

df_dbg = df.sort_values(["mmsi", "date_time_utc"]).copy()

g = df_dbg.groupby("mmsi")
df_dbg["prev_time"] = g["date_time_utc"].shift()
df_dbg["prev_lat"] = g["lat"].shift()
df_dbg["prev_lon"] = g["lon"].shift()
df_dbg["dt"] = (df_dbg["date_time_utc"] - df_dbg["prev_time"]).dt.total_seconds()

df_dbg["dist"], df_dbg["implied_speed_ms"] = haversine(
    df_dbg["prev_lat"], df_dbg["prev_lon"],
    df_dbg["lat"], df_dbg["lon"],
    df_dbg["dt"]
)

df_dbg[["date_time_utc", "lat", "lon", "speed", "dt", "dist", "implied_speed_ms", "trajectory_id"]]
df_dbg.to_csv("checkkkyyy.csv", index=False)

print(df["trajectory_id"].unique())

print(df["mmsi"].nunique())
df = df.loc[df["mmsi"] >= 250000000].copy()


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

#print(haversine(61.4800, 1.80169, 61.48145, 1.80270, 1))
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