import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/Results_GECIquant_somas.csv')
df_md = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/Results_GECIquant_microdomains.csv')

#print(df)

soma_list = pd.unique(df['ROI Label'])
soma_intensities = [[] for _ in range(len(soma_list))]

md_list = pd.unique(df_md['ROI Label'])
md_intensities = [[] for _ in range(len(md_list))]

for i in range(len(df)):
    for j in range(len(soma_list)):
        if df["ROI Label"][i] == soma_list[j]:
            soma_intensities[j].append(df["Intensity"][i])

for i in range(len(df_md)):
    for j in range(len(md_list)):
        if df_md["ROI Label"][i] == md_list[j]:
            md_intensities[j].append(df_md["Intensity"][i])

fig, ax = plt.subplots(len(soma_intensities))
for i in range(len(soma_intensities)):
    ax[i].plot(soma_intensities[i])
    ax[i].set_ylabel(soma_list[i], fontsize=9)
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].tick_params(labelbottom=False)
    plt.xlabel('Frame number', fontsize=9)

# fig, ax1 = plt.subplots(len(md_intensities), figsize=(30, 15))
# for i in range(len(md_intensities)):
#     ax1[i].plot(md_intensities[i])
#     #ax1[i].set_ylabel('Fluorescence', fontsize=9)
#     #plt.xlabel('Frame number', fontsize=9)

plt.show()


#plotting microdomains in multiple groups of subplots

# num_plots_per_fig = 15
# num_figs = len(md_intensities)//num_plots_per_fig + (len(md_intensities) % num_plots_per_fig != 0)
# for fig_num in range(num_figs):
#     start = fig_num * num_plots_per_fig
#     end = min((fig_num + 1) * num_plots_per_fig, len(md_intensities))
#     fig, ax1 = plt.subplots(end-start, figsize=(10, 10))
#     for i in range(start, end):
#         ax1[i-start].plot(md_intensities[i])
#         ax1[i-start].set_ylabel(md_list[i], fontsize=9)
#         ax1[i-start].tick_params(labelbottom=False)
#         ax1[i-start].spines[['right', 'top']].set_visible(False)
#         plt.xlabel('Frame number', fontsize=9)
#     fig.tight_layout()
#     plt.show()








