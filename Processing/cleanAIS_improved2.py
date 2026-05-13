import os
from time import time
import pandas as pd
import numpy as np
import math

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

def extract_trajectories2(df, max_gap="120min", max_speed=20):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["mmsi", "date_time_utc"])

    g = df.groupby("mmsi", sort=False)

    df["prev_time"] = g["date_time_utc"].shift(1)
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_lon"] = g["lon"].shift(1)

    df["dt"] = (df["date_time_utc"] - df["prev_time"]).dt.total_seconds()

    df["dist_to_prev"], df["speed_to_prev"] = haversine(
        df["prev_lat"], df["prev_lon"],
        df["lat"], df["lon"],
        df["dt"]
    )

    max_gap_s = pd.Timedelta(max_gap).total_seconds()

    break_edge = (
        df["dt"].isna()
        | (df["dt"] <= 0)
        | (df["dt"] > max_gap_s)
        | (df["speed_to_prev"] > max_speed)
    )

    df["segment_id"] = break_edge.groupby(df["mmsi"]).cumsum() - 1

    df["trajectory_id"] = (
        df["mmsi"].astype(str) + "-" + df["segment_id"].astype(str)
    )

    return df.drop(columns=[
        "prev_time", "prev_lat", "prev_lon",
        "dt", "dist_to_prev", "speed_to_prev", "segment_id"
    ])

def remove_spike_fragments(df, max_messages=5, min_jump_m=500):
    seg = (df.groupby(["mmsi", "trajectory_id"])
             .agg(start_time=("date_time_utc", "min"),
                  start_lat=("lat", "first"),
                  start_lon=("lon", "first"),
                  n=("lat", "size"))
             .reset_index()
             .sort_values(["mmsi", "start_time"]))

    g = seg.groupby("mmsi", sort=False)
    seg["prev_lat"] = g["start_lat"].shift(1)
    seg["prev_lon"] = g["start_lon"].shift(1)
    jump_m, _ = haversine(seg["prev_lat"], seg["prev_lon"],
                          seg["start_lat"], seg["start_lon"], 1)
    drop = (seg["n"] < max_messages) & (jump_m > min_jump_m)
    bad_ids = set(seg.loc[drop, "trajectory_id"])
    return df[~df["trajectory_id"].isin(bad_ids)]

def reconnect_subtracks(df, max_gap="120min", max_speed=20):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["mmsi", "date_time_utc"])

    # Get start/end of each current subtrack
    seg = (
        df.groupby(["mmsi", "trajectory_id"])
          .agg(
              start_time=("date_time_utc", "min"),
              end_time=("date_time_utc", "max"),
              start_lat=("lat", "first"),
              start_lon=("lon", "first"),
              end_lat=("lat", "last"),
              end_lon=("lon", "last"),
              n=("lat", "size"),
          )
          .reset_index()
          .sort_values(["mmsi", "start_time"])
    )

    g = seg.groupby("mmsi", sort=False)

    seg["prev_end_time"] = g["end_time"].shift(1)
    seg["prev_end_lat"] = g["end_lat"].shift(1)
    seg["prev_end_lon"] = g["end_lon"].shift(1)

    seg["gap_s"] = (seg["start_time"] - seg["prev_end_time"]).dt.total_seconds()
    seg["gap_dist_m"], seg["gap_speed_ms"] = haversine(
        seg["prev_end_lat"], seg["prev_end_lon"],
        seg["start_lat"], seg["start_lon"], seg["gap_s"]
    )

    max_gap_s = pd.Timedelta(max_gap).total_seconds()

    new_group = (
        seg["gap_s"].isna()
        | (seg["gap_s"] <= 0)
        | (seg["gap_s"] > max_gap_s)
        | (seg["gap_speed_ms"] > max_speed)
    )

    seg["joined_id"] = new_group.groupby(seg["mmsi"]).cumsum()

    seg["new_trajectory_id"] = (
        seg["mmsi"].astype(str) + "-" + seg["joined_id"].astype(str)
    )

    id_map = seg.set_index("trajectory_id")["new_trajectory_id"]

    df["trajectory_id"] = df["trajectory_id"].map(id_map)

    return df.sort_values(["trajectory_id", "date_time_utc"])

def remove_duplicate_positions(df):
    print("Removing duplicate positions per trajectory")

    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    before = len(df)

    df = (
        df.sort_values(["trajectory_id", "date_time_utc"])
          .drop_duplicates(subset=["trajectory_id", "lat", "lon"], keep="first")
    )

    removed = before - len(df)
    print(f"Removed {removed:,} duplicate-position rows")

    return df

def remove_trajectories_few_instances(df, min_instances):
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


def remove_spikes_three_point(df,
                              perp_ratio_threshold=0.5, min_perp=5.0,
                              path_ratio_threshold=3.0, min_excursion=100.0):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"])

    g = df.groupby("trajectory_id", sort=False)
    lat_prev = g["lat"].shift(1);  lon_prev = g["lon"].shift(1)
    lat_next = g["lat"].shift(-1); lon_next = g["lon"].shift(-1)

    d_ab, _ = haversine(lat_prev, lon_prev, df["lat"], df["lon"], 1)
    d_bc, _ = haversine(df["lat"], df["lon"], lat_next, lon_next, 1)
    d_ac, _ = haversine(lat_prev, lon_prev, lat_next, lon_next, 1)

    d_ac_safe = d_ac.replace(0, np.nan)

    # Test 1: perpendicular offset, Heron -> perpendicular distance from B to line AC
    s = 0.5 * (d_ab + d_bc + d_ac)
    area = np.sqrt(np.clip(s * (s - d_ab) * (s - d_bc) * (s - d_ac), 0, None))
    perp_dist = (2 * area) / d_ac_safe
    off_axis = (perp_dist / d_ac_safe > perp_ratio_threshold) & (perp_dist > min_perp)

    # Test 2: detour vs direct path, comparing dist AB and BC with dist AC. If AB AND BC >> AC -> spike
    path_ratio = (d_ab + d_bc) / d_ac_safe
    out_and_back = (path_ratio > path_ratio_threshold) & ((d_ab + d_bc) > min_excursion)

    spike = (off_axis | out_and_back).fillna(False)

    print(f"Removed {spike.sum():,} three-point spikes "
          f"(off-axis: {off_axis.fillna(False).sum():,}, "
          f"out-and-back: {out_and_back.fillna(False).sum():,})")

    return df.loc[~spike]                               

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
    #df = remove_spike_fragments(df)
    #df = remove_trajectories_few_instances(df, min_instances=5)
    #df = reconnect_subtracks(df)
    df = remove_duplicate_positions(df)
    df = remove_trajectories_w_low_avg_speed(df)
    df = remove_short_trajectories(df)
    df = remove_trajectories_few_instances(df, min_instances=100)
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
    print(haversine(67.980330, 14.838186, 67.982895, 14.846458, 1))
   
    """  df = pd.read_parquet("Processed_AIS_2024/Concatenated/01.parquet")

    mmsis = df["mmsi"].drop_duplicates().head(10)
    df_small = df[df["mmsi"].isin(mmsis)].copy()

    df_small = all(df_small)
    df_small.to_parquet("Processed_AIS_2024/Cleaned/01_2024_testy.parquet", index=False) """

    # NEW CLEANING
    # this one works very good! preserves the most data! with the old cleaning we lost a lot of data due to very strict acceleration filter!
