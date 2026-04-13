import pandas as pd
import os
import numpy as np
import pyarrow.parquet as pq

TYPE_CODE_FISHING_VESSEL = 30 # AIS code for fishing vessels, filtering out fishing vessels

REGION_LAT = 55 # We want all vessels north of 62 degrees north
REGION_LON_EAST = 45
REGION_LON_WEST = -10

def readParquetFile(filename):
    df = pd.read_parquet(filename, engine='pyarrow')
    return df

def readParquetFile_onlyFish(filename):
    table = pq.read_table(
        filename,
        filters=[
            ("ship_type", "==", 30),
            ("lat", ">=", REGION_LAT),
            ("lon", ">=", REGION_LON_WEST),
            ("lon", "<=", REGION_LON_EAST),
        ]
    )
    return table.to_pandas()

def readParquetFile_onlySTS(filename, sts_filename):
    sts = pd.read_csv("Data/fangstdata_2024_sts.csv")
    radio_giver = sts["Radiokallesignal (seddel)"].unique()
    radio_receiver = sts["Mottakende fartøy rkal"].unique()
    giver_receiver = np.concatenate((radio_giver, radio_receiver))

    table = pq.read_table(
        filename,
        columns=["callsign", "lon", "lat", "date_time_utc", "ship_type"],
        filters=[("callsign", "in", giver_receiver)]
    )
    sts_ais = table.to_pandas()
  
    return sts_ais

def extractFishingVessels(df):
    print("Extracting fishing vessels based on type code: ", TYPE_CODE_FISHING_VESSEL)
    fishingVessels = df.loc[df["ship_type"] == TYPE_CODE_FISHING_VESSEL].copy()
    return fishingVessels

def filterRegion(df):
    print("Filtering out all vessels that are not within the region of interest.")

    #df.drop("geometry_wkt", axis=1, inplace=True) # Delete the geoemtry_wkt column

    insideRegion = df.loc[(df["lat"] >= REGION_LAT) & (df["lon"] >= REGION_LON_WEST) & (df["lon"] <= REGION_LON_EAST)]

    return insideRegion

def saveToCSV(filename, df):
    df.to_csv(filename)
    print("Succesfully saved to: ", filename)

def saveToParquet(outfile, df):
    df.to_parquet(outfile, engine="pyarrow", compression="snappy")
    print("Successfully saved to ", outfile)


def readFilterSave(filename, saveFilename):
    df = readParquetFile_onlyFish(filename)
    #fishingVessels = extractFishingVessels(df)
    #insideRegion = filterRegion(df)
    #saveToCSV(saveFilename, insideRegion)
    saveToParquet(saveFilename, df)

def readFilterSave2(filename, saveFilename, sts_filename):
    df = readParquetFile_onlySTS(filename, sts_filename)
    #fishingVessels = extractFishingVessels(df)
    insideRegion = filterRegion(df)
    #saveToCSV(saveFilename, insideRegion)
    saveToParquet(saveFilename, insideRegion)


def resample(df, step="1min"):
    print("Resampling")
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(by=["mmsi", "date_time_utc"])
    df = df.set_index("date_time_utc")
    #print(df.head())

    resampled = (
        df.groupby("mmsi", group_keys=False)
          .apply(lambda g: g.resample(step, origin=g.index.min()).first())
    )

    try:

        resampled["mmsi"] = resampled["mmsi"].astype("int64")
    except:
        print("Nah")
    #print(df.head())
    return resampled


