import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#from data_loader import load_data

from load_data import load_data_

df, df_features, df_mea, df_sw_events = load_data_()  # Replace 0 with the desired subfolder_choice

from SPW_plot import process_spw

#df, df_features, df_mea, df_sw_events = load_data() # load all returned values from load_data function

process_spw(df_mea, df_sw_events)

#print(f"The mean time difference between calcium events before sharp wave events is {mean_time_difference_before} seconds.")
#print(f"The mean time difference between calcium events after sharp wave events is {mean_time_difference_after} seconds.")


#plot_transients(df) # plot transients using filtered events
