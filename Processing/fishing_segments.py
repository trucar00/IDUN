import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH = "Processed_AIS_2024/Resampled_3h/0160minfill.csv"

df_full = pd.read_csv(PATH, usecols=["mmsi", "callsign", "date_time_utc", "lon", "lat", "speed", "cog"])
#df_full = df_full.dropna(subset=["lon", "lat", "speed", "cog"])

print(df_full["date_time_utc"].head())

first_20_mmsi = df_full["mmsi"].drop_duplicates().tail(50)

# Keep only those vessels
df = df_full[df_full["mmsi"].isin(first_20_mmsi)].copy()

def angle_wrap(a):
    
    return (a + 180) % 360 - 180

df["del_cog"] = df.groupby("mmsi")["cog"].diff().apply(angle_wrap)
df["del_cog"] = df["del_cog"].fillna(0)
df["is_steaming"] = 0

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

window_length = pd.Timedelta(hours=1)

""" for mmsi, d in df.groupby("mmsi"):
        d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
        d = d.sort_values("date_time_utc").reset_index(drop=True)
        start = d["date_time_utc"].min()
        end = d["date_time_utc"].max()
        current = start
        print(type(current))

        while current + window_length <= end:
            window_df = d[(d["date_time_utc"] >= current)
                         & (d["date_time_utc"] < (current + window_length))].copy()
            window_avg_speed = window_df["speed"].mean()
            window_std_speed = window_df["speed"].std()
            delcog_activity = np.mean(np.abs(window_df["del_cog"]) > 0.2)
            
            current += window_length
            print(window_df.head()) """

steaming = []
fishing = []

for mmsi, idx in tqdm(df.groupby("mmsi").groups.items()):

    d = df.loc[idx].copy()
    start = d["date_time_utc"].min()
    end = d["date_time_utc"].max()
    current = start

    while current + window_length <= end:

        mask = (
            (df.index.isin(idx)) &
            (df["date_time_utc"] >= current) &
            (df["date_time_utc"] < current + window_length)
        )

        window_df_all = df.loc[mask]  # includes NaNs
        window_df_clean = window_df_all.dropna(subset=["lon", "lat", "speed", "cog"])

        if len(window_df_clean) < 5:
            current += window_length
            continue

        window_avg_speed = window_df_clean["speed"].mean()
        window_std_speed = window_df_clean["speed"].std()
        delcog_activity = np.mean(np.abs(window_df_clean["del_cog"]) > 5)
        #print(delcog_activity)

        if (
            (window_avg_speed > 8) and
            (window_std_speed < 2) and
            (delcog_activity < 0.20)   # <-- see note below
        ):
            df.loc[mask, "is_steaming"] = 1
            #steaming.append(window_df_all)
        #else:
            #fishing.append(window_df_all)
        
        #print(window_df_clean.head())
        current += window_length


#steaming_df = pd.concat(steaming)
#fishing_df = pd.concat(fishing)


#df.to_csv("fishing01.csv")
fig, ax = plt.subplots(figsize=(10,8))


for mms, d in df.groupby("mmsi"):
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")

    # Keep only steaming coordinates
    lon_plot = d["lon"].where(d["is_steaming"] == 1)
    lat_plot = d["lat"].where(d["is_steaming"] == 1)

    ax.plot(lon_plot, lat_plot, linewidth=1, alpha=0.7)

plt.title("Steaming")
plt.show()

fig, ax = plt.subplots(figsize=(10,8))

for mms, d in df.groupby("mmsi"):
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")

    # Keep only steaming coordinates
    lon_plot = d["lon"].where(d["is_steaming"] == 0)
    lat_plot = d["lat"].where(d["is_steaming"] == 0)

    ax.plot(lon_plot, lat_plot, linewidth=1, alpha=0.7)

plt.title("Fishing")
plt.show()

# looks very ugly because we plot all with steaming = 1 -> creates gaps where fishing is "detected" plots between these gaps creating big jumps. Same for fishing
# need to divide into trajectories, steaming trajectory, fishing trajectory. then it will make more sense.
# then try for the trawling and autoline only datasets. see if we can remove all steaming trajectories. 