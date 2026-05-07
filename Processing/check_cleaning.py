import pandas as pd
import matplotlib.pyplot as plt

new_df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_mix.parquet")
old_df = pd.read_parquet("Processed_AIS_2024/Cleaned/01_old_clean.parquet")

new_df["date_time_utc"] = pd.to_datetime(new_df["date_time_utc"])
old_df["date_time_utc"] = pd.to_datetime(old_df["date_time_utc"])


def plot_mmsi(mmsi):
    new_dd = new_df[new_df["mmsi"] == mmsi]
    old_dd = old_df[old_df["mmsi"] == mmsi]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 7), sharex=True, sharey=True
    )

    # OLD
    for traj_id, d in old_dd.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc")
        ax1.plot(d["lon"], d["lat"], ".", markersize=2)

    ax1.set_title("Old cleaning")

    # NEW
    for traj_id, d in new_dd.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc")
        ax2.plot(d["lon"], d["lat"], ".", markersize=2)

    ax2.set_title("New cleaning")

    # Equal aspect ratio (important for geo plots)
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    plt.suptitle(f"MMSI: {mmsi}")
    plt.tight_layout()
    plt.show()


# Example: plot one MMSI


for mmsi, d in new_df.groupby("mmsi"):
    plot_mmsi(mmsi)