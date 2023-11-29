import os
import csv
import pandas as pd
##
def load_data_():
    data_folder_path = r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data'

    # list subfolders
    subfolders = sorted([f.name for f in os.scandir(data_folder_path) if f.is_dir()])

    while True:
        print("Please choose a subfolder:")
        for i, subfolder in enumerate(subfolders):
            print(f"{i + 1}: {subfolder}")
        subfolder_choice = int(input()) - 1
        if subfolder_choice in range(len(subfolders)):
            break
        else:
            print("Invalid choice. Please try again.")

    chosen_subfolder_path = os.path.join(data_folder_path, subfolders[subfolder_choice])
    chosen_subfolder = subfolders[subfolder_choice]
    print(chosen_subfolder)
    # list files in subfolder
    files = sorted([f.name for f in os.scandir(chosen_subfolder_path) if f.is_file() and not f.name.startswith('.')])

    # choose first 4 numeric files
    numeric_files = sorted([f for f in files if f.split('_')[0].isdigit()])
    chosen_files = [os.path.join(chosen_subfolder_path, numeric_files[i]) for i in range(4)]

    # load data from chosen files
    df = pd.read_csv(chosen_files[0])
    df_features = pd.read_csv(chosen_files[1])
    df_mea = pd.read_csv(chosen_files[2])
    df_sw_events = pd.read_csv(chosen_files[3])

    df = df.T  # shifts columns and rows
    indexes = []
    for index_row in df_features:
        indexes.append(index_row)

    # replace column names (e.g. Event 3)  with only numbers (3)
    for i in range(len(df.columns)):
        df.rename(columns={df.columns[i]: str(i + 1)}, inplace=True)

    # iterate over the column indexes and drop any columns that are not in the list
    for col_name in df.columns:
        if col_name not in indexes:
            df = df.drop(col_name, axis=1)

    # df2 = df.iloc[:, 175:185]

    df = df.drop(df.index[-1])
    df = df.drop(df.index[0])

    col_order = list(df.iloc[1].argsort())

    # rearrange the columns of the dataframe based on the order of the second row

    print(df)

    return df, df_features, df_mea, df_sw_events
#load_data()

