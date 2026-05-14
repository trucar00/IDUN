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
        df.sort_values(["mmsi", "date_time_utc"])
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

def remove_stationary(df, speed_threshold=0.5, min_duration="10min"):
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


def remove_impossible_jumps(df, max_speed=20.0):
    """
    Iteratively remove points causing impossible jumps in each trajectory.

    Find the first edge (a, b) where the implied speed a→b exceeds max_speed:
      - If it is the very first edge of the trajectory (a is point 0), drop a.
        Either side could be wrong, and the rest of the trajectory is our
        best evidence about which side is real — so we bet on the rest.
      - Otherwise drop b. The chain up to a has already been validated, so
        b is the new arrival that broke the rule.
    Repeat until no impossible jumps remain.
    """
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = (df.sort_values(["mmsi", "trajectory_id", "date_time_utc"])
            .reset_index(drop=True))

    keep = np.ones(len(df), dtype=bool)
    n_dropped = 0

    for _, group in df.groupby(["mmsi", "trajectory_id"], sort=False):
        lats = group["lat"].to_numpy()
        lons = group["lon"].to_numpy()
        times = group["date_time_utc"].to_numpy()
        positions = group.index.to_numpy()
        active = list(range(len(group)))

        i = 1
        while i < len(active):
            a, b = active[i - 1], active[i]
            dt_s = (times[b] - times[a]) / np.timedelta64(1, "s")

            if dt_s > 0:
                _, speed = haversine(lats[a], lons[a], lats[b], lons[b], dt_s)
                bad = speed > max_speed
            else:
                bad = True  # duplicate/inverted timestamp

            if not bad:
                i += 1
                continue

            if i == 1:
                keep[positions[a]] = False  # drop first point
                active.pop(0)
            else:
                keep[positions[b]] = False  # drop later point
                active.pop(i)
            n_dropped += 1
            # don't advance i — recheck the new pair at this position

    print(f"Removed {n_dropped:,} impossible-jump points")
    return df.loc[keep].copy().reset_index(drop=True)


def remove_duplicate_positions(df):
    print("Removing duplicate positions per trajectory")

    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    before = len(df)

    df = (
        df.sort_values(["trajectory_id", "date_time_utc"])
          .drop_duplicates(subset=["trajectory_id", "lat", "lon"], keep="last")
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
    df = remove_duplicate_positions(df)
    df = remove_impossible_jumps(df)

    while True:
        before = len(df)
        df = remove_spikes_three_point(df)
        print(before, "->", len(df))
        if len(df) == before:
            break

    df = remove_trajectories_w_low_avg_speed(df)
    df = remove_trajectories_few_instances(df, min_instances=100)
    df = reindex_trajectory_ids(df)
    df = df.drop(columns=["dsrc", "imo", "ship_type", "maneuvre", "geom", "rot", "true_heading", "draught", "geometry_wkt"], errors="ignore")
    return df

def main(months, concat_path, cleaned_path):
    start = time()

    for month in range(1,months+1):
        getfile = f"{concat_path}{month:02d}.parquet"
        savefile = f"{cleaned_path}{month:02d}.parquet"
        if os.path.exists(getfile):
            print("Cleaning up: ", getfile)
            df = pd.read_parquet(getfile, engine="pyarrow")
            df = all(df)
            df.to_parquet(savefile, engine="pyarrow", compression="snappy")
            print("Saved cleaned data to: ", savefile)          
        else:
            print("Missing: ", getfile)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")

    return

if __name__ == "__main__":
    #main()
    #print(haversine(67.980330, 14.838186, 67.982895, 14.846458, 1))
   
    df = pd.read_parquet("Processed_AIS_2024/Concatenated/01.parquet")

    mmsis = df["mmsi"].drop_duplicates().head(10)
    df_small = df[df["mmsi"].isin(mmsis)].copy()

    df_small = all(df_small)
    df_small.to_parquet("Processed_AIS_2024/Cleaned/01_2024_clean.parquet", index=False)

    # NEW CLEANING
    # this one works very good! preserves the most data! with the old cleaning we lost a lot of data due to very strict acceleration filter!
