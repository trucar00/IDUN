from pathlib import Path
from Processing import getData, concatParquets, cleanAIS, downSample #, buildTrainingSets, makeh5
#from Model import autoencoder, latentSpacePlot, clusterFunc

# Need folder paths, anc creation of these folders if they dont exist?

# Define the time period we are interested in
YEAR = 2024
MONTHS = 12
DAYS = 30

TRAJECTORY_LENGTH = 1 #hours
RESAMPLE_STEP = "1min"

AIS_PATH = f"{YEAR}/date_utc={YEAR}"
FILTERED_PATH = f"Processed_AIS_{YEAR}/Parquets/"
CONCAT_PATH = f"Processed_AIS_{YEAR}/Concatenated/"

GEAR_PATH = f"Processed_AIS_{YEAR}/gear_specific/not_feb_2024.csv"
CLEAN_PATH = f"Processed_AIS_{YEAR}/Cleaned/"

RESAMPLE_PATH = f"Processed_AIS_{YEAR}/Resampled/"
TRAINING_SETS_PATH = f"Training_sets_{YEAR}/{TRAJECTORY_LENGTH}h/"
TRAJECTORIES_PATH = f"{TRAINING_SETS_PATH}trajectories.h5"
AUTOENCODER_PATH = f"Model/{TRAJECTORY_LENGTH}h/"
CLUSTER_PATH = f"clusters_{TRAJECTORY_LENGTH}h.json"
#LATENT_PATH = f"Model/{TRAJECTORY_LENGTH}h/"

def main():
    folder_paths = [FILTERED_PATH, CONCAT_PATH, CLEAN_PATH, RESAMPLE_PATH, TRAINING_SETS_PATH]

    for p in folder_paths:
        path = Path(p)

        if path.exists():
            print(f"[EXISTS]  {path}")
        else:
            path.mkdir(parents=True)
            print(f"[CREATED] {path}")

    #getData.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH) #choose nr of days
    
    getData.main2(months=MONTHS, filtered_path=FILTERED_PATH) # all days in each month
    #concatParquets.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)
    
    concatParquets.main2(months=MONTHS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)
    cleanAIS.main(months=MONTHS, concat_path=CONCAT_PATH, cleaned_path=CLEAN_PATH, traj_length=TRAJECTORY_LENGTH)
    downSample.main(cleaned_path=CLEAN_PATH, resampled_path=RESAMPLE_PATH, step=RESAMPLE_STEP, months=MONTHS)

    #buildTrainingSets.main(resampled_path=RESAMPLE_PATH, training_path=TRAINING_SETS_PATH, months=MONTHS, traj_length=TRAJECTORY_LENGTH)
    #makeh5.createh5(training_path=TRAINING_SETS_PATH, traj_length=TRAJECTORY_LENGTH, trajectories_path=TRAJECTORIES_PATH)
    #autoencoder.main(trajectories_path=TRAJECTORIES_PATH, ae_path=AUTOENCODER_PATH, traj_length=TRAJECTORY_LENGTH)
    #latentSpacePlot.main(latent_space_path=AUTOENCODER_PATH, featureset_path=TRAINING_SETS_PATH)
    #clusterFunc.main(latent_space_path=AUTOENCODER_PATH, featureset_path=TRAINING_SETS_PATH, clusters_path=CLUSTER_PATH, saveClusters=True)
    #cluster_plot


if __name__ == "__main__":
    main()