import os
import pandas as pd

def load_data():
    # define path for the data folder
    data_folder_path = r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data'

    # get list of files in data folder and sort based on the number in the filename
    data_files = sorted(os.listdir(data_folder_path), key=lambda x: int(x.split('_')[1]))

    print("Data files:")
    for i, file in enumerate(data_files):
        print(f"{i + 1}. {file}")

    while True:
        file_nums = input("Enter the file numbers you want to load for fluo and features separated by comma: ")
        try:
            fluo_num, features_num = map(int, file_nums.strip().split(','))
            if not (1 <= fluo_num <= len(data_files)) or not (1 <= features_num <= len(data_files)):
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter valid file numbers separated by comma.")

    # load data from csv files
    fluo_file = os.path.join(data_folder_path, data_files[fluo_num - 1])
    features_file = os.path.join(data_folder_path, data_files[features_num - 1])
    df = pd.read_csv(fluo_file, index_col='Event Index')
    df_features = pd.read_csv(features_file)

    return df, df_features