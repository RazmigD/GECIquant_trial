import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_mea = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/mea_data/11_32_37_trace.csv')
df_sw_events = spw_detection = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/mea_data/11_32_37_SWevents.csv')

df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/calcium_data/11_32_37_slice1_1.csv')
df_features = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/calcium_data/11_32_37_slice1_2.csv')

#
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
col_order = list(df.iloc[1].argsort())

# rearrange the columns of the dataframe based on the order of the second row
df = df.iloc[:, col_order]

print(df)

# ------------------------------------------------------------------------------------------------------

# #Check calcium events between sharp wave event onset and end
#
# import numpy as np
# import pandas as pd
#
# # assume df and spw_detection are defined and loaded with data
#
# calcium_onsets = (df.iloc[1,:])/3
# calcium_ends = (df.iloc[2,:])/3
# calcium_peaks = (calcium_onsets + calcium_ends) / 2
# sharp_wave_onsets = (spw_detection.iloc[:, 0])/1000
# sharp_wave_ends = (spw_detection.iloc[:, 1])/1000
# spw_freq = 100/len(spw_detection))
#
# # Initialize counters for events before and after sharp waves
#
# events = 0
# peaks = 0
# no_onset = 0
# no_peak = 0
#
# for i, sharp_wave_onset in enumerate(sharp_wave_onsets):
#     sharp_wave_end = sharp_wave_ends[i]
#     calcium_peaks_within_spw = [x for x in calcium_peaks if x >= sharp_wave_onset and x <= sharp_wave_end]
#
#     if not calcium_peaks_within_spw:
#         no_peak += 1
#         continue
#     for calcium_peak in calcium_peaks_within_spw:
#         peaks += 1
#
# # Print out counts of events before and after sharp waves
# print(f"{peaks} calcium events peaks occurred within sharp wave events.")
# print(f"No calcium event peak detected within sharp wave event window {no_peak} times.")


# ------------------------------------------------------------------------------------------------------


# Check calcium events before and after sharp wave events

calcium_onsets = (df.iloc[1,:])/3
sharp_wave_onsets = (spw_detection.iloc[:, 0])/1000
spw_freq = len(spw_detection)/100

# Initialize counters for events before and after sharp waves
events_before = []
events_after = []
time_differences = []
no_event = 0

for sharp_wave_onset in sharp_wave_onsets:
    calcium_onsets_within_2_sec = [x for x in calcium_onsets if x >= sharp_wave_onset - spw_freq and x <= sharp_wave_onset + spw_freq]
    if not calcium_onsets_within_2_sec:
        no_event += 1
        #print("No calcium event detected within 2 seconds of sharp wave event.")
        continue
    for calcium_onset in calcium_onsets_within_2_sec:
        time_difference = calcium_onset - sharp_wave_onset
        time_differences.append(time_difference)

        if time_difference < 0:
            events_before.append(time_difference*(-1))
            #print(f"Calcium event occurred {abs(time_difference):.2f} seconds before sharp wave event.")
        elif time_difference > 0:
            events_after.append(time_difference)
            #print(f"Calcium event occurred {time_difference:.2f} seconds after sharp wave event.")
        #else:
            #print("Calcium event occurred at the same time as sharp wave event.")

events_before = np.array(events_before)
events_after = np.array(events_after)

print( )
print("Time difference for events before SPW onset", list(events_before))
print("Time difference for events after SPW onset", list(events_after))
print( )

mean_time_difference_before = np.mean(events_before)
mean_time_difference_after = np.mean(events_after)

# Print out counts of events before and after sharp waves
print(f"{len(events_before)} calcium events occurred before sharp wave events.")
print(f"{len(events_after)} calcium events occurred after sharp wave events.")
print(f"No calcium events detected {no_event} times before or after SPW")
print( )

print(f"The mean time difference between calcium events before sharp wave events is {mean_time_difference_before} seconds.")
print(f"The mean time difference between calcium events after sharp wave events is {mean_time_difference_after} seconds.")


# Create a histogram of the time differences
plt.hist(time_differences, bins=20)
plt.xlabel('Time difference (s)')
plt.ylabel('Frequency')
plt.title('Distribution of time differences between calcium and sharp wave events')
plt.show()

# ------------------------------------------------------------------------------------------------------

# #Plot Sharp Wave Trace

#Define the x-values for the vertical lines


from data_loader import load_data
#, _, df_mea, df_sw_events = load_data()


import matplotlib.pyplot as plt
import numpy as np

x_values_green = df_sw_events.iloc[:, 0]
x_values_red = df_sw_events.iloc[:, 1]

# Plot the trace data
fig, ax = plt.subplots()

ax.plot(df_mea.iloc[:, 1])
ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
# Set the x-axis limits to zoom in on a specific region
#plt.xlim(25000, 35000)

# Draw the green and red vertical lines
for x in x_values_green:
    plt.axvline(x=x, color='green', linewidth=0.3)

for x in x_values_red:
    plt.axvline(x=x, color='red', linewidth=0.3)

# Show the plot
plt.show()

# ------------------------------------------------------------------------------------------------------

# # #Plotting the average Sharp Wave Trace
# # Initialize empty list to store sharp wave event traces
# sharp_wave_traces = []
#
# # Loop through each sharp wave event in spw_detection
# for onset, end in zip(df_sw_events.iloc[:, 0], df_sw_events.iloc[:, 1]):
# # Slice the filtered trace using onset and end points
#     sharp_wave_trace = df_mea.iloc[onset:end, 1].values
# # Append the sharp wave event trace to the list
#     sharp_wave_traces.append(sharp_wave_trace)
#
# # Determine the maximum length of all the sharp wave event traces
# max_length = max([len(trace) for trace in sharp_wave_traces])
#
# # Pad the shorter traces with zeros to ensure that they all have the same length
# padded_traces = [np.pad(trace, (0, max_length - len(trace))) for trace in sharp_wave_traces]
#
# # Compute mean of sharp wave event traces to obtain average sharp wave trace
# average_sharp_wave_trace = sum(padded_traces) / len(padded_traces)
#
# # Plot the average sharp wave trace
# plt.plot(average_sharp_wave_trace)
# plt.show()

#return average_sharp_wave_trace


