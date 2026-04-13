import pandas as pd
from time import time
from pathlib import Path

# --- Concatenates the filtered parquet files ---

def main(months, days, filtered_path, concat_path):
    start = time()
    print("Concatenating first five days of each month into common parquet file.")
    for month in range(1, months+1):
        dfs = []
        for day in range(1, days+1):
            filepath = f"{filtered_path}{month:02d}-{day:02d}.parquet"
            print(f"Concat of 2024-{month:02d}-{day:02d}")
            df = pd.read_parquet(filepath, engine="pyarrow")
            dfs.append(df)

        concat_df = pd.concat(dfs, ignore_index=True)

        concat_df.to_parquet(f"{concat_path}{month:02d}.parquet")

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")

def main2(months, filtered_path, concat_path):
    start = time()
    print("Concatenating daily parquet files into monthly parquet files.")

    filtered_path = Path(filtered_path)
    concat_path = Path(concat_path)

    for month in range(1, months + 1):
        # Matches: 01-01.parquet, 01-02.parquet, ...
        files = sorted(filtered_path.glob(f"{month:02d}-*.parquet"))

        if not files:
            print(f"No files found for month {month:02d}")
            continue

        print(f"Month {month:02d}: {len(files)} files")

        dfs = []
        for fp in files:
            print(f"Reading {fp.name}")
            dfs.append(pd.read_parquet(fp, engine="pyarrow"))

        concat_df = pd.concat(dfs, ignore_index=True)
        out_fp = concat_path / f"{month:02d}.parquet"
        concat_df.to_parquet(out_fp, engine="pyarrow", index=False)

    end = time()
    print("Done! It took:", (end - start) / 60, "minutes.")

if __name__ == "__main__":
    main()