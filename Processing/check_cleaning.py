import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

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

new_df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_new_clean_2023.parquet")
old_df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_old_clean_2023.parquet")
#old_df = pd.read_csv("Processed_AIS_2024/Cleaned/01_old_clean_2023.csv")
new_df = new_df.loc[new_df["mmsi"] >= 250000000].copy()
old_df = old_df.loc[old_df["mmsi"] >= 250000000].copy()
print("new: ", new_df.shape, new_df["mmsi"].nunique(), " old: ", old_df.shape, old_df["mmsi"].nunique())
print(new_df.columns, old_df.columns)

new_df["date_time_utc"] = pd.to_datetime(new_df["date_time_utc"])
old_df["date_time_utc"] = pd.to_datetime(old_df["date_time_utc"])

g = new_df.groupby("trajectory_id", sort=False)
new_df["prev_lat"] = g["lat"].shift(1)
new_df["prev_lon"] = g["lon"].shift(1)

new_df["prev_time"] = g["date_time_utc"].shift(1)
new_df["dt"] = (new_df["date_time_utc"] - new_df["prev_time"]).dt.total_seconds()

new_df["dist_to_prev"], new_df["speed_to_prev"] = haversine(
    new_df["prev_lat"], new_df["prev_lon"],
    new_df["lat"], new_df["lon"],
    new_df["dt"]
)

cols = [
    "mmsi", "trajectory_id", "date_time_utc",
    "lat", "lon", "prev_lat", "prev_lon",
    "dt", "dist_to_prev", "speed_to_prev"
]

new_df = new_df.sort_values(["trajectory_id", "date_time_utc"]).reset_index(drop=True)

bad = new_df["speed_to_prev"] > 20

bad_with_neighbors = new_df[
    bad
    | bad.groupby(new_df["trajectory_id"]).shift(1, fill_value=False)
    | bad.groupby(new_df["trajectory_id"]).shift(-1, fill_value=False)
].copy()

bad_with_neighbors["is_high_speed"] = bad_with_neighbors["speed_to_prev"] > 20

print(bad_with_neighbors[cols + ["is_high_speed"]].to_string(index=True))

traj_id = "257437000-4"

start_time = pd.Timestamp("2024-01-16 22:17:00")
end_time   = pd.Timestamp("2024-01-16 22:20:00")

subset = (
    new_df[
        (new_df["mmsi"] == 257437000) &
        (new_df["date_time_utc"] >= start_time) &
        (new_df["date_time_utc"] <= end_time)
    ]
    .sort_values("date_time_utc")
)

cols = [
    "date_time_utc",
    "trajectory_id",
    "lat",
    "lon",
    "dt",
    "dist_to_prev",
    "speed_to_prev"
]

#print(subset[cols].to_string(index=True))

fig, ax = plt.subplots(figsize=(8, 8))
for traj, d in subset.groupby("trajectory_id"):
    print(traj)
    ax.scatter(d["lon"], d["lat"])

plt.scatter(3.806065, 60.768960, color="red")
plt.show()

def plot_mmsi(mmsi):
    new_dd = new_df[new_df["mmsi"] == mmsi]
    old_dd = old_df[old_df["mmsi"] == mmsi]
    print("NEW: ", new_dd["trajectory_id"].nunique())
    print("OLD: ", old_dd["trajectory_id"].nunique())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7, 4), sharex=True, sharey=True
    )

    # OLD
    for traj_id, d in old_dd.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc")
        ax1.plot(d["lon"], d["lat"], ".", markersize=2)

        if not d.empty:
            call = d["callsign"].iloc[0]
        else:
            call = None  # or handle it differently


    ax1.set_title("Old cleaning")

    # NEW
    for traj_id, d in new_dd.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc")
        # plot trajectory
        ax2.plot(d["lon"], d["lat"], ".", markersize=2)

        # highlight suspicious points
        high_speed = d[d["speed_to_prev"] > 20]

        ax2.scatter(
            high_speed["lon"],
            high_speed["lat"],
            color="red",
            s=20,
            label="speed > 20"
        )

    ax2.set_title("New cleaning")

    plt.suptitle(f"MMSI: {mmsi}")
    plt.tight_layout()
    plt.show()


# Example: plot one MMSI


for mmsi, d in new_df.groupby("mmsi"):
    plot_mmsi(mmsi)