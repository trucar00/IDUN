import os
from . import dataProcessing
from time import time
import glob

# --- Gets the copy of NTNUs AIS-data from Kystverket ---
# --- readFilterSave() reads the parquet files, and filters out fishing vessels within region ---

def main(months, days, filtered_path):
    start = time()
    print("Getting data from NTNUs copy of AIS-data from Kystverket.")
    for month in range(1,months+1):
        for day in range(1, days+1):
            filepath = f"~Data-preMaster/2024/date_utc=2024-{month:02d}-{day:02d}"
            if os.path.exists(filepath):
                for entry in os.scandir(filepath):
                    if entry.is_file() and entry.name.endswith(".parquet"):
                        print("Processing file: ", entry.path)
                        dataProcessing.readFilterSave(entry.path, f"{filtered_path}{month:02d}-{day:02d}.parquet")
                        
            else:
                print("Missing: ", filepath)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")

def main2(months, filtered_path):
    start = time()
    print("Getting data from NTNUs copy of AIS-data from Kystverket.")
    for month in range(1,months+1):
        pattern = f"../../Data-preMaster/2024/date_utc=2024-{month:02d}-*" # ADD Y: for running locally
        folders = sorted(glob.glob(pattern))
        if not folders:
            print("No folders for month:", month)
            continue
        day = 0
        for folder in folders:
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.endswith(".parquet"):
                    print("Processing file: ", entry.path)
                    day += 1
                    dataProcessing.readFilterSave(entry.path, f"{filtered_path}{month:02d}-{day:02d}.parquet") #use readFilterSave2 for STS

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")


if __name__ == "__main__":
    main(months=12, days=5, filtered_path="2024")
