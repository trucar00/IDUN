import os
from time import time
import pandas as pd
import numpy as np
import math

# NOTE: the original 3h dataset was created with a min_duration = 10 min not 15 min as for the 6h. In the cleaned version, we use 15 min as standard.

# HELPER

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

# -------

def remove_duplicate_timestamps(df):
    print("Removing duplicate timestamps per MMSI")

    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    before = len(df)

    df = (
        df.sort_values(["mmsi", "date_time_utc", "speed"], ascending=[True, True, False])
          .drop_duplicates(subset=["mmsi", "date_time_utc"], keep="first")
    )

    removed = before - len(df)
    print(f"Removed {removed:,} duplicate-timestamp rows")

    return df

def remove_invalid(df, min_cog=0, max_cog=360, min_speed=0, max_speed=30):
    print("Removing invalid rows")

    # Ensure numeric columns
    for col in ["cog", "speed", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a mask for valid values
    valid_mask = (
        df["cog"].between(min_cog, max_cog, inclusive="both")
        & df["speed"].between(min_speed, max_speed, inclusive="both")
    )

    invalid_count = len(df) - valid_mask.sum()
    print(f"Removed {invalid_count:,} invalid rows")

    return df[valid_mask]

def remove_stationary(df, speed_threshold=0.5, min_duration="15min"):
    print("Removing stationary")

    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["mmsi", "date_time_utc"])

    df["stationary"] = df["speed"] < speed_threshold

    # Group changes in stationary state PER MMSI
    df["grp"] = (
        df.groupby("mmsi")["stationary"]
          .apply(lambda s: (s != s.shift()).cumsum())
          .reset_index(level=0, drop=True)
    )

    drop_idx = []

    for (_, _), g in df[df["stationary"]].groupby(["mmsi", "grp"]):
        duration = g["date_time_utc"].max() - g["date_time_utc"].min()
        if duration >= pd.Timedelta(min_duration):
            drop_idx.append(g.index)

    if drop_idx:
        df = df.drop(np.concatenate(drop_idx))

    return df.drop(columns=["stationary", "grp"])

def extract_trajectories(df, time_threshold="60min"):
    df = df.sort_values(["mmsi", "date_time_utc"])
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["dt"] = df.groupby("mmsi")["date_time_utc"].diff().dt.total_seconds()
    tt = pd.Timedelta(time_threshold).total_seconds()

    df["traj_id"] = (df["dt"] > tt).groupby(df["mmsi"]).cumsum()
    df["trajectory_id"] = df["mmsi"].astype(str) + "-" + df["traj_id"].astype(str)

    return df.drop(columns=["dt", "traj_id"])

def remove_trajectories_few_instances(df, min_instances=100):
    print(f"Removing trajectories with fewer than {min_instances} messages")

    counts = df["trajectory_id"].value_counts()
    valid_traj = counts[counts >= min_instances].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj)]

    removed = len(counts) - len(valid_traj)
    print(f"Removed {removed} trajectories")

    return df_filtered

def remove_short_trajectories(df, traj_length=30):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    durations = (
    df.groupby("trajectory_id")["date_time_utc"]
      .agg(["min", "max"])
      .assign(duration=lambda x: x["max"] - x["min"])
    )

    valid_traj_ids = durations[durations["duration"] >= pd.Timedelta(minutes=traj_length)].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj_ids)]
    print("Before removing short trajectories:", df["trajectory_id"].nunique())
    print("After:", df_filtered["trajectory_id"].nunique())

    return df_filtered

def remove_trajectories_w_low_avg_speed(df, min_avg_speed_knots=1):
    avg_speed = df.groupby("trajectory_id")["speed"].mean()

    stationary_traj_ids = avg_speed[avg_speed < min_avg_speed_knots].index

    df = df[~df["trajectory_id"].isin(stationary_traj_ids)].copy()

    print(f"Removed {len(stationary_traj_ids)} trajectories with avg speed < {min_avg_speed_knots} knots")
    print(f"Remaining trajectories: {df['trajectory_id'].nunique()}")
    
    return df


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

    spike = ((ratio > ratio_threshold) & (perp_dist > min_perp)).fillna(False)

    print(f"Removed {spike.sum():,} three-point spikes")
    return df.loc[~spike].drop(columns=["lat_prev", "lon_prev",
                                        "lat_next", "lon_next"],
                               errors="ignore")

def reindex_trajectory_ids(df):
    print("Reindexing trajectory IDs")

    df = df.sort_values(["mmsi", "date_time_utc"])

    # Map old trajectory IDs to new sequential ones per MMSI
    new_ids = []
    for mmsi, group in df.groupby("mmsi"):
        unique_trajs = {old_id: new_id for new_id, old_id in enumerate(sorted(group["trajectory_id"].unique()))}
        new_ids.append(group.assign(
            trajectory_id_new=group["trajectory_id"].map(unique_trajs),
            trajectory_id=lambda g: g["mmsi"].astype(str) + "-" + g["trajectory_id_new"].astype(str)
        ))

    df = pd.concat(new_ids, ignore_index=True)
    df = df.drop(columns=["trajectory_id_new"])
    return df


def all(df):
    df = remove_duplicate_timestamps(df)
    df = remove_invalid(df)
    df = remove_stationary(df)
    df = extract_trajectories(df)
    df = remove_trajectories_w_low_avg_speed(df)
    df = remove_short_trajectories(df)
    df = remove_trajectories_few_instances(df)
    df = remove_spikes_three_point(df)
    df = reindex_trajectory_ids(df)
    df = df.drop(columns=["dsrc", "imo", "ship_type", "maneuvre", "geom", "status", "rot", "true_heading", "length", "draught", "ais_class", "geometry_wkt"])
    return df

def main(months, concat_path, cleaned_path):
    start = time()

    for month in range(1,months+1):
        getfile = f"{concat_path}{month:02d}.parquet"
        savefile = f"{cleaned_path}{month:02d}.parquet" # Remove
        #savefile = f"{cleaned_path}.csv"
        if os.path.exists(getfile):
            print("Cleaning up: ", getfile)
            df = pd.read_parquet(getfile)
            df = all(df)
            #df.to_csv(savefile, index=False)
            df.to_parquet(savefile, engine="pyarrow", compression="snappy")
            print("Saved cleaned data to: ", savefile)          
        else:
            print("Missing: ", getfile)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")

    return

if __name__ == "__main__":
    #main()
    df = pd.read_parquet("Processed_AIS_2024/Concatenated/01.parquet")
    print(df.columns)

    mmsis = df["mmsi"].drop_duplicates().head(25)
    df_small = df[df["mmsi"].isin(mmsis)].copy()

    df_small = all(df_small)
    df_small.to_parquet("Processed_AIS_2024/Cleaned/01_mix.parquet", index=False)

    # NEW CLEANING
    # this one works very good! preserves the most data! with the old cleaning we lost a lot of data due to very strict acceleration filter!
