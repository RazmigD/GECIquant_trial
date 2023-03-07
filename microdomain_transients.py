import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

df_md = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/Results_GECIquant_microdomains.csv')

md_list = pd.unique(df_md['ROI Label'])
md_intensities = [[] for _ in range(len(md_list))]

for i in range(len(df_md)):
    for j in range(len(md_list)):
        if df_md["ROI Label"][i] == md_list[j]:
            md_intensities[j].append(df_md["Intensity"][i])

num_plots_per_fig = 15
num_figs = len(md_intensities)//num_plots_per_fig + (len(md_intensities) % num_plots_per_fig != 0)
for fig_num in range(num_figs):
    start = fig_num * num_plots_per_fig
    end = min((fig_num + 1) * num_plots_per_fig, len(md_intensities))
    fig, ax1 = plt.subplots(end-start, figsize=(10, 10))
    for i in range(start, end):
        ax1[i-start].plot(md_intensities[i])
        ax1[i-start].set_ylabel(md_list[i], fontsize=9)
        ax1[i-start].tick_params(labelbottom=False)
        ax1[i-start].spines[['right', 'top']].set_visible(False)
        plt.xlabel('Frame number', fontsize=9)
    fig.tight_layout()
    plt.show()


















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