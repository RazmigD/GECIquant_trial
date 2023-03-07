import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/Aqua_Curve_Output.csv', index_col='Event Index')
df = df.T #shifts columns and rows

df_area = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/Aqua_Output_Excel.csv')
areas = df_area.loc[0].values #Extract area values and put them in a list

#print(len(areas))

#pdb.set_trace()


areas = [value for value in areas if isinstance(value, (int, float)) and not np.isnan(value)] #Removing NaN or non num values
df.loc['Event_Area'] = areas #Appending areas list to df: each event has its own area
#print(df)

#pdb.set_trace()

df1 = df.iloc[:, 700:] #df1 is generated to test plotting
df2 = df1.copy() #df2 is just df1

for col in df2.columns: #Checks area (last value) of each event (column) and drops event not corresponding to defined area value
    if df2[col].iloc[-1] < 100: #SET AREA HERE
        df2.drop(col, axis=1, inplace=True)

print(df2)

#pdb.set_trace()

fig, ax = plt.subplots(len(df2.columns), 1, figsize=(15, 10))
for i in range(len(df2.columns)):
    start_frame = int(df2.iloc[0, i])
    end_frame = int(df2.iloc[1, i])

    ax[i].plot(df2.iloc[2:-1, i]) #For each column, skip 1st 2 and last rows
    ax[i].axvspan(start_frame, end_frame, color='yellow', alpha=0.5) #Highlight the transients

    # ax[i].set_xticks(range(0, len(df1.iloc[2:, i]), 20))  # Display labels every 20 frames
    # ax[i].set_xticklabels(range(1, len(df1.iloc[2:, i]), 20))  # Set the custom labels for the x axis

    ax[i].set_xticks([start_frame, end_frame]) # Specify position of x ticks
    ax[i].set_xticklabels([start_frame, end_frame], fontsize = 7) #Specify label of x ticks
    ax[i].set_ylabel(df2.columns[i])
    ax[i].text(1.05, 0.95, f"Area = {df2.iloc[-1, i]}",
               fontsize=8,
               horizontalalignment='right',
               verticalalignment='top',
               transform=ax[i].transAxes)
    ax[i].spines[['right', 'left', 'top', 'bottom']].set_visible(False)
fig.tight_layout()
plt.show()




#print(df_area_)


# To plot all events, to be checked LATER

# n_cols = 1
# n_rows = int(np.ceil(len(df.columns) / n_cols))

# fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 10),
#                        sharex=True, sharey=False)
# ax = ax.flatten()
#
# for i in range(len(df.columns)):
#     start_frame = int(df.iloc[0, i])
#     end_frame = int(df.iloc[1, i])
#     ax[i].plot(df.iloc[2:, i])
#     ax[i].axvspan(start_frame, end_frame, color='yellow', alpha=0.5)
#
#     ax[i].set_xticks([start_frame, end_frame])  # Specify position of x ticks
#     ax[i].set_xticklabels([start_frame, end_frame], fontsize=7)  # Specify label of x ticks
#     ax[i].set_ylabel(df.columns[i])
#     ax[i].spines[['right', 'left', 'top', 'bottom']].set_visible(False)
#
# for i in range(len(df.columns), n_rows * n_cols):
#     ax[i].remove()
#
# fig.tight_layout()
# plt.show()



















# md_list = pd.unique(df_md['ROI Label'])
# md_intensities = [[] for _ in range(len(md_list))]
#
# for i in range(len(df_md)):
#     for j in range(len(md_list)):
#         if df_md["ROI Label"][i] == md_list[j]:
#             md_intensities[j].append(df_md["Intensity"][i])
#
# list_indices_to_plot = [102, 124, 130]
#
# fig, ax2 = plt.subplots(len(list_indices_to_plot), figsize=(10, 10))
#
# for j, i in enumerate(list_indices_to_plot): #j takes the count i takes the value of my list
#     ax2[j].plot(md_intensities[i])
#     ax2[j].set_ylabel('Fluorescence', fontsize=9)
#     ax2[j].set_xlabel('Frame number', fontsize=9)
#     ax2[j].spines[['right', 'top']].set_visible(False)
#     ax2[j].tick_params(labelbottom=False)
#
# fig.tight_layout()
# plt.show()




