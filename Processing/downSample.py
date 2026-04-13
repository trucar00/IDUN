import pandas as pd
from time import time
import os
import numpy as np

# --- Code for downsampling AIS data ---

MAX_INTERP_GAP = pd.Timedelta("60min")

def downsample2(df, step):
    df = df.copy()

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).set_index("date_time_utc")

    theta = np.deg2rad(df["cog"].astype(float))
    df["cog_x"] = np.cos(theta)
    df["cog_y"] = np.sin(theta)

    def resample_and_interpolate(g):
        traj = g.name  # <-- this group's trajectory_id string

         # regular grid from first to last timestamp of this trajectory
        regular_index = pd.date_range(
            start=g.index.min(),
            end=g.index.max(),
            freq=step
        )

        # union original timestamps + target grid, so interpolation uses real points
        g_union = g.reindex(g.index.union(regular_index)).sort_index()

        # linear spatial interpolation in time
        interp_cols = [c for c in ["lon", "lat", "speed", "cog_x", "cog_y"] if c in g_union.columns]
        g_union[interp_cols] = g_union[interp_cols].interpolate(
            method="time",
            limit_area="inside"
        )

        # keep only the regular timestamps
        g_res = g_union.loc[regular_index].copy()

        r = np.hypot(g_res["cog_x"], g_res["cog_y"])
        g_res["cog_x"] = g_res["cog_x"] / r
        g_res["cog_y"] = g_res["cog_y"] / r
        g_res["theta"] = np.arctan2(g_res["cog_y"], g_res["cog_x"])
        g_res["cog_interp"] = np.rad2deg(g_res["theta"]) % 360

        # Fill identifiers that exist
        id_cols = [c for c in ["mmsi", "callsign", "ship_name"] if c in g_res.columns]
        if id_cols:
            g_res[id_cols] = g_res[id_cols].ffill().bfill()

        # Re-add trajectory_id as a string column
        g_res["trajectory_id"] = traj

        return g_res

    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(resample_and_interpolate)
          .reset_index()
          .rename(columns={"index": "datetime"})
    )

    # Ensure trajectory_id is string dtype
    resampled["trajectory_id"] = resampled["trajectory_id"].astype("string")
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    return resampled.drop(columns=["cog", "theta", "cog_x", "cog_y"])

def downsample(df, step):
    df = df.copy()

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).set_index("date_time_utc")

    def resample_and_interpolate(g):
        traj = g.name  # <-- this group's trajectory_id string

        g_res = g.resample(step, origin=g.index.min()).first()

        # Interpolate only continuous signals that exist
        interp_cols = [c for c in ["lon", "lat", "speed", "cog"] if c in g_res.columns]
        if interp_cols:
            g_res[interp_cols] = g_res[interp_cols].interpolate("linear")

        # Fill identifiers that exist
        id_cols = [c for c in ["mmsi", "callsign"] if c in g_res.columns]
        if id_cols:
            g_res[id_cols] = g_res[id_cols].ffill().bfill()

        # Re-add trajectory_id as a string column
        g_res["trajectory_id"] = traj

        return g_res

    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(resample_and_interpolate)
          .reset_index()
    )

    # Ensure trajectory_id is string dtype
    resampled["trajectory_id"] = resampled["trajectory_id"].astype("string")
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    return resampled


def main(cleaned_path, resampled_path, step, months):
    start = time()

    for month in range(1, months+1):
        getfile = f"{cleaned_path}{month:02d}.csv"
        savefile = f"{resampled_path}{month:02d}.csv"
        if os.path.exists(getfile):
            print("Resampling: ", getfile)
            df = pd.read_csv(getfile)
            df = downsample2(df, step)
            df.to_csv(savefile, index=False)
            print(f"Saved resampled data for 2024-{month:02d} to {savefile}")          
        else:
            print("Missing: ", getfile)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    return

if __name__ == "__main__":
    #main()
    print("Already created resampled csv's")