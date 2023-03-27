import pdb
import numpy as np

from transients_plotter import plot_transients
from transients_plotter import get_filtered_events
from data_loader import load_data
from SPW_plot import process_spw

df, df_features, df_mea, df_sw_events = load_data() # load all returned values from load_data function

df = get_filtered_events(df, df_features)

plot_transients(df) # plot transients using filtered events
process_spw(df_mea, df_sw_events)

### STILL NOT PLOTTING OUTPUT OF PROCESS_SPW TRY TO FIX THIS ISSUE!!!

#average_sharp_wave_trace = process_spw(df_mea, df_sw_events)


#process_spw(df_mea, df_sw_events)




