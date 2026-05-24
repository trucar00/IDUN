from pathlib import Path
from Processing import getData, concatParquets, cleanAIS_improved2, downSample #, buildTrainingSets, makeh5
#from Model import autoencoder, latentSpacePlot, clusterFunc

# Need folder paths, anc creation of these folders if they dont exist?

# Define the time period we are interested in
YEAR = 2025
MONTHS = 12
DAYS = 30

TRAJECTORY_LENGTH = 2 #hours
RESAMPLE_STEP = "1min"

AIS_PATH = f"{YEAR}/date_utc={YEAR}"
FILTERED_PATH = f"Processed_AIS_{YEAR}/Parquets/"
CONCAT_PATH = f"Processed_AIS_{YEAR}/Concatenated/"

GEAR_PATH = f"Processed_AIS_{YEAR}/gear_specific/not_feb_2024.csv"
CLEAN_PATH = f"Processed_AIS_{YEAR}/Cleaned_pq_new/"

RESAMPLE_PATH = f"Processed_AIS_{YEAR}/Resampled/"
TRAINING_SETS_PATH = f"Training_sets_{YEAR}/{TRAJECTORY_LENGTH}h/"
TRAJECTORIES_PATH = f"{TRAINING_SETS_PATH}trajectories.h5"
AUTOENCODER_PATH = f"Model/{TRAJECTORY_LENGTH}h/"
CLUSTER_PATH = f"clusters_{TRAJECTORY_LENGTH}h.json"
#LATENT_PATH = f"Model/{TRAJECTORY_LENGTH}h/"

def main():
    folder_paths = [FILTERED_PATH, CONCAT_PATH, CLEAN_PATH, RESAMPLE_PATH]

    for p in folder_paths:
        path = Path(p)

        if path.exists():
            print(f"[EXISTS]  {path}")
        else:
            path.mkdir(parents=True)
            print(f"[CREATED] {path}")

    #getData.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH) #choose nr of days
    
    #getData.main3(months=MONTHS, filtered_path=FILTERED_PATH, year=YEAR) # all days in each month
    #concatParquets.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)
    
    concatParquets.main2(months=MONTHS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)
    #cleanAIS_improved2.main(months=MONTHS, concat_path=CONCAT_PATH, cleaned_path=CLEAN_PATH)
    #downSample.main(cleaned_path=CLEAN_PATH, resampled_path=RESAMPLE_PATH, step=RESAMPLE_STEP, months=MONTHS)



if __name__ == "__main__":
    main()