import os
import pandas as pd

def load_data():
    data_folder_path = r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/calcium_data'
    data_mea_folder_path = r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/mea_data'

    # get list of files in data folder and sort based on the number in the filename
    data_files = sorted([f for f in os.listdir(data_folder_path) if not f.startswith('.')],
                        key=lambda x: int(x.split('_')[1]))
    data_mea_files = sorted([f for f in os.listdir(data_mea_folder_path) if not f.startswith('.')],
                            key=lambda x: int(x.split('_')[1]))

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

    print("MEA files:")
    for i, file in enumerate(data_mea_files):
        print(f"{i + 1}. {file}")

    while True:
        file_nums_ = input("Enter the file numbers you want to load for MEA trace and SW events separated by comma: ")
        try:
            mea_trace, sw_events = map(int, file_nums_.strip().split(','))
            if not (1 <= mea_trace <= len(data_mea_files)) or not (1 <= sw_events <= len(data_mea_files)):
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter valid file numbers separated by comma.")

    # load data from csv files
    mea_file = os.path.join(data_mea_folder_path, data_mea_files[mea_trace - 1])
    sw_file = os.path.join(data_mea_folder_path, data_mea_files[sw_events - 1])
    df_mea = pd.read_csv(mea_file)
    df_sw_events = pd.read_csv(sw_file)

    return df, df_features, df_mea, df_sw_events