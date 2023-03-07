import pandas as pd
from transients_plotter import plot_transients
from transients_plotter import get_filtered_events
from data_loader import load_data

df, df_features = load_data()

df = get_filtered_events(df, df_features)
plot_transients(df)






