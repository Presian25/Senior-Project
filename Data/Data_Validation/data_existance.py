"""
The goal of this file is to ensure all of the following for an x dataset:
1) All timestamps from the start to the last exist
2) No duplicate rows
3) No missing values on rows

This script is used for all assets' datasets but crypto,
as it is a continuous market, while the others are not.

Substitute the output file path to validate different datasets
"""

import pandas as pd
from pathlib import Path

DAYS = 729
YEARS = 2
TYPES = ["_1h", "_1d"]
FREQ = "B"
INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets" 
CONTINUOUS_MARKET = "Crypto"

#Method to identify whether the analyzed dataset is on hourly or on daily data
def identify_dataset_type(filename):
    for type in TYPES: 
        if type in filename:
            print(f"Dataset type: {type}")
            return type
    return None



#Method to check if all daily data entries exist in the dataset
def data_completeness(df, filename, type):
    time_range = pd.date_range(start = df.index.min(), end = df.index.max(), freq = FREQ)
    df = df.sort_index()

    if type == "_1d":
        missing = time_range.difference(df.index)
        if len(missing) > 0:
            print(f"The dataset {filename} has missing {len(missing)} entries")
            return False
        print(f"The dataset {filename} has no missing days")
        return True

    elif type =="_1h":
        days = pd.to_datetime(pd.unique(df.index.date))
        missing = time_range.difference(days)

        if len(missing) > 0:
            print(f"The dataset {filename} has missing {len(missing)} entries")
            return False

        gaps = {}
        for day, group in df.groupby(df.index.date):
            normal_gaps = group.index.to_series().diff().dropna()
            big_gaps = normal_gaps[normal_gaps > pd.Timedelta(hours=1)]
            if len(big_gaps) > 0: 
                gaps[day] = list(big_gaps)

        if len(gaps) > 0:
            print(f"The dataset {filename} has  {len(gaps)} hourly gaps")
            return False
        return True
    else:
        print("File cannot be classified as hourly/daily")
        return True

#Method to ensure no duplicate entries in the datasets
def ensure_unqiueness(df, filename):
    duplicates = df.index[df.index.duplicated()]
    if len(duplicates) > 0: 
        print(f"There exist {len(duplicates)} number of duplicated entries")
        return False
    print("No duplicate entries in the dataset")
    return True


def main():

    csv_files = [
        csv for csv in sorted(INPUT.rglob("*.csv"))
        if CONTINUOUS_MARKET not in csv.parts
    ]
    print(f"Found {len(csv_files)} files to validate.\n")

    for file in csv_files:
        filename = file.name
        print(f"--- {filename} ---")

        df = pd.read_csv(file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()]
        type = identify_dataset_type(filename)

        data_completeness(df, filename, type)
        ensure_unqiueness(df, filename)
        print()


if __name__ == "__main__":
    main()
        






