
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from data_loader import load_data
df, df_features, _, _ = load_data()

# getting indexes of filtered events
def get_filtered_events(df, df_features):
    df = df.T #shifts columns and rows
    indexes = []
    for index_row in df_features:
        indexes.append(index_row)

    # replace column names (e.g. Event 3)  with only numbers (3)
    for i in range(len(df.columns)):
        df.rename(columns={df.columns[i]: str(i+1)}, inplace=True)

    # iterate over the column indexes and drop any columns that are not in the list
    for col_name in df.columns:
        if col_name not in indexes:
            df = df.drop(col_name, axis=1)

    #df2 = df.iloc[:, 175:185]

    df = df.drop(df.index[-1])

    areas = df_features.loc[0].values  # Extract area values and put them in a list
    areas = [value for value in areas if isinstance(value, (int, float)) and not np.isnan(value)]  # Removing NaN or non num values
    df.loc['Event_Area'] = areas  # Appending areas list to df: each event has its own area

    col_order = list(df.iloc[0].argsort())

    # rearrange the columns of the dataframe based on the order of the second row
    df = df.iloc[:, col_order]

    return df

# get the order of the columns based on the values in the second row (start frame)
#col_order = list(df.iloc[0].argsort())

# rearrange the columns of the dataframe based on the order of the second row
#df = df.iloc[:, col_order]

#pdb.set_trace()

#df2 = df.iloc[:, 60:70] #df1 is generated to test plotting

#pdb.set_trace()

def plot_transients(df):
    n_cols = len(df.columns)
    group_size = 10
    n_groups = n_cols // group_size + int(n_cols % group_size != 0)

    for group_idx in range(n_groups):
        start_col = group_idx * group_size
        end_col = min((group_idx + 1) * group_size, n_cols)
        group_cols = df.iloc[:, start_col:end_col]

        amplitudes = []
        fig, ax = plt.subplots(len(group_cols.columns), 1, figsize=(15, 10))
        for i in range(len(group_cols.columns)):
            start_frame = int(group_cols.iloc[0, i])
            end_frame = int(group_cols.iloc[1, i])

            signal = group_cols.iloc[2:-1, i]  # Extract signal for current event

            baseline = signal[:start_frame].mean() # Find baseline of signal
            signal = signal - baseline  # Subtract baseline from signal
            peak = signal.max() # Find peak of signal
            amplitude = peak - baseline # Calculate amplitude of signal
            amplitudes.append(round(amplitude,2))  # Appending amplitudes

            ax[i].plot(signal) # For each column, skip 1st 2 and last rows

            ax[i].axvspan(start_frame, end_frame, color='yellow', alpha=0.2) #Highlight the transients
            ax[i].set_xticks([start_frame, end_frame]) # Specify position of x ticks
            ax[i].set_xticklabels([round(start_frame/3), round(end_frame/3)], fontsize = 4) #Specify label of x ticks
            ax[i].set_ylabel(group_cols.columns[i])
            ax[i].text(1.05, 0.95, f"Am = {amplitudes[i]}",
                       fontsize=8,
                       horizontalalignment='right',
                       verticalalignment='top',
                       transform=ax[i].transAxes)
            ax[i].text(1.05, 0.7, f"D = {round((end_frame - start_frame) * 0.324)}s",
                       fontsize=8,
                       horizontalalignment='right',
                       verticalalignment='top',
                       transform=ax[i].transAxes)
            ax[i].text(1.05, 0.43, f"Ar = {round(df.iloc[-1, i])} \u00B5m\u00B2 ",
                       horizontalalignment='right',
                       verticalalignment='top',
                       fontsize=8,
                       transform=ax[i].transAxes)
            ax[i].spines[['right', 'left', 'top', 'bottom']].set_visible(False)
        fig.tight_layout()
        plt.show()

    with open('amplitudes.pkl', 'wb') as f:
        pickle.dump(amplitudes, f)





