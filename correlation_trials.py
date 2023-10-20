import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

df_mea = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0003_11_10_08_trace.csv')
df_sw_events = spw_detection = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0004_11_10_08_SWevents.csv')

df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0001_11_10_08_slice1.csv')
df_features = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0002_11_10_08_slice2.csv')

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
df = df.drop(df.index[0])

col_order = list(df.iloc[1].argsort())
#
# # rearrange the columns of the dataframe based on the order of the second row
#
# print(df)

# ------------------------------------------------------------------------------------------------------

#Check calcium events between sharp wave event onset and end

# import numpy as np
# import pandas as pd
#
# # assume df and spw_detection are defined and loaded with data

sw_durations = df_sw_events.iloc[:, 2] / 1000  # SW event durations

avg_sw_duration = sw_durations.mean()

sharp_wave_onsets = (df_sw_events.iloc[:, 0])/1000
sharp_wave_ends = (df_sw_events.iloc[:, 1])/1000
spw_freq = len(df_sw_events)/100




print('Average SW duration', avg_sw_duration)

calcium_onsets = (df.iloc[0,:]) /3

calcium_onsets = calcium_onsets.tolist()
print(calcium_onsets)
print(len(calcium_onsets))
col_indexes = df.columns.values

col_indexes = col_indexes.tolist()


print('column indexes:', col_indexes)
print(len(col_indexes))
my_dict = {col_indexes[i]: calcium_onsets[i] for i in range(len(calcium_onsets))}
print(my_dict)



print(df)


#print("Calcium Onsets", list(calcium_onsets))
print('dataframe length:', len(df.columns))




print(sharp_wave_onsets)

events_before = []
events_within = []
events_after = []

for sharp_wave_onset in sharp_wave_onsets:
    calcium_onsets_b4_after = [x for x in my_dict.values() if x >= sharp_wave_onset - avg_sw_duration and x <= sharp_wave_onset + avg_sw_duration]
    print(f"sharp_wave_onset: {sharp_wave_onset}, calcium_onsets_b4_after: {calcium_onsets_b4_after}")

    for k, v in my_dict.items():
        if v in calcium_onsets_b4_after and v < sharp_wave_onset:
            print(f"Appending {k}: {v} to events_before")
            events_before.append({k: v})
        elif v in calcium_onsets_b4_after and v > sharp_wave_onset:
            print(f"Appending {k}: {v} to events_within")
            events_within.append({k: v})

for sharp_wave_end in sharp_wave_ends:
    calcium_onsets_post_end = [x for x in my_dict.values() if x >= sharp_wave_end and x <= sharp_wave_end + avg_sw_duration]
    print(f"sharp_wave_end: {sharp_wave_end}, calcium_onsets_post_end: {calcium_onsets_post_end}")

    for k, v in my_dict.items():
        if v in calcium_onsets_post_end:
            print(f"Appending {k}: {v} to events_after")
            events_after.append({k: v})

events_between = []

for i in range(len(sharp_wave_onsets)-1):
    start = sharp_wave_ends[i] + avg_sw_duration
    end = sharp_wave_onsets[i+1] - avg_sw_duration
    calcium_events_between = [x for x in my_dict.values() if start <= x <= end]

    for k, v in my_dict.items():
        if v in calcium_events_between:
            print(f"Appending {k}: {v} to events_between")
            events_between.append({k: v})

print()
print('Events before SW onset:', events_before)
prop_before = (len(events_before)/len(my_dict))*100
print('Proportion:',prop_before)

print('Events within SW events:', events_within)
prop_within = (len(events_within)/len(my_dict))*100
print('Proportion:',prop_within)

print('Events after SW end:', events_after)
prop_after = (len(events_after)/len(my_dict))*100
print('Proportion:', prop_after)

print('Events between SW events:', events_between)
prop_between = (len(events_between)/len(my_dict))*100
print('Proportion:', prop_between)

print(len(df_features.columns))
print(len(df_sw_events)/100)



calcium_ends = (df.iloc[1,:])/3
#print("Calcium Ends", list(calcium_ends))


df_summary = pd.read_csv('df_summary.csv')

print(df_summary)
import pickle
with open('SPW_frequencies.pkl', 'rb') as f:
    SPW_frequencies = pickle.load(f)

with open('ca_event_num.pkl', 'rb') as f:
    ca_event_num = pickle.load(f)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

calcium_event_num = [197, 372, 221, 219, 306, 217, 128, 70, 43]
swr_freq = [0.67, 0.47, 0.78, 0.78, 0.49, 1.08, 0.81, 0.79, 0.93]

# Perform linear regression
regression_line = np.polyfit(calcium_event_num, swr_freq, 1)
line_fn = np.poly1d(regression_line)

# Scatter plot
plt.scatter(calcium_event_num, swr_freq, color='blue', label='Data Points')

# Regression line
plt.plot(calcium_event_num, line_fn(calcium_event_num), color='red', linestyle='-', label='Regression Line')

plt.xlabel('Calcium Event Number')
plt.ylabel('SWR Frequency')
plt.title('Relationship between Calcium Event Number and SWR Frequency')

plt.legend()

# Remove upper and right axes spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


import scipy.stats as stats

SPW_frequency = [0.67, 0.47, 0.78, 0.78, 0.49, 1.08, 0.81, 0.79, 0.93]
calcium_events = [197, 372, 221, 219, 306, 217, 128, 70, 43]

correlation_coeff, p_value = stats.pearsonr(SPW_frequency, calcium_events)

plt.text(min(SPW_frequency), max(calcium_events), f"Correlation: {correlation_coeff:.2f}\np-value: {p_value:.2f}",
         fontsize=10, color='black', verticalalignment='top', horizontalalignment='left')

print("Correlation coefficient:", correlation_coeff)
print("p-value:", p_value)

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

#
# # Check calcium events before and after sharp wave events
#
# calcium_onsets = (df.iloc[0, :]) / 3


# spw_freq = len(spw_detection)/100
#
# # Initialize counters for events before and after sharp waves
# events_before = []
# events_after = []
# time_differences = []
# no_event = 0
#
# for sharp_wave_onset in sharp_wave_onsets:
#     calcium_onsets_within_2_sec = [x for x in calcium_onsets if x >= sharp_wave_onset - spw_freq and x <= sharp_wave_onset + spw_freq]
#     if not calcium_onsets_within_2_sec:
#         no_event += 1
#         #print("No calcium event detected within 2 seconds of sharp wave event.")
#         continue
#     for calcium_onset in calcium_onsets_within_2_sec:
#         time_difference = calcium_onset - sharp_wave_onset
#         time_differences.append(time_difference)
#
#         if time_difference < 0:
#             events_before.append(time_difference*(-1))
#             #print(f"Calcium event occurred {abs(time_difference):.2f} seconds before sharp wave event.")
#         elif time_difference > 0:
#             events_after.append(time_difference)
#             #print(f"Calcium event occurred {time_difference:.2f} seconds after sharp wave event.")
#         #else:
#             #print("Calcium event occurred at the same time as sharp wave event.")
#
# events_before = np.array(events_before)
# events_after = np.array(events_after)
#
# #pdb.set_trace()
# print('Calcium onsets:', calcium_onsets)
#
#
#
# print("Time difference for events before SPW onset", list(events_before))
# print("Time difference for events after SPW onset", list(events_after))
# print( )
#
# mean_time_difference_before = np.mean(events_before)
# mean_time_difference_after = np.mean(events_after)
#
# # Print out counts of events before and after sharp waves
# print(f"{len(events_before)} calcium events occurred before sharp wave events.")
# print(f"{len(events_after)} calcium events occurred after sharp wave events.")
# print(f"No calcium events detected {no_event} times before or after SPW")
# print( )
#
# print(f"The mean time difference between calcium events before sharp wave events is {mean_time_difference_before} seconds.")
# print(f"The mean time difference between calcium events after sharp wave events is {mean_time_difference_after} seconds.")
#
#
# # Create a histogram of the time differences
# plt.hist(time_differences, bins=20)
# plt.xlabel('Time difference (s)')
# plt.ylabel('Frequency')
# plt.title('Distribution of time differences between calcium and sharp wave events')
# plt.show()

# ------------------------------------------------------------------------------------------------------

# #Plot Sharp Wave Trace

#Define the x-values for the vertical lines


from data_loader import load_data
#, _, df_mea, df_sw_events = load_data()
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# if len(df_mea) > 97100:
#     df_sw_events.iloc[:, [0, 1, 3]] = (df_sw_events.iloc[:, [0, 1, 3]] - 2380) / 1000
#
# # Print the updated DataFrame
# print(df_sw_events)
#
# # Get the values for the electrode trace
# trace_values = df_mea.iloc[:, 1].values
#
# # Subtract 2380 from the trace values
# trace_values_adj = trace_values[2380:] - trace_values[2380]
#
# # Get the adjusted x-axis values
# x_values_adj = np.arange(0, len(trace_values_adj))
#
# # Subtract 2380 from the start and end points of the sharp waves
# x_values_green_adj = df_sw_events.iloc[:, 0].values
# x_values_red_adj = df_sw_events.iloc[:, 1].values
#
# # Plot the trace data with the adjusted x-axis and y-axis values
# fig, ax = plt.subplots()
# ax.plot(x_values_adj, trace_values_adj)
#
# # Draw the green and red vertical lines with adjusted x-axis values
#
#
# # Set the x-axis label
# ax.set_xlabel('Time (ms)')
#
# # Hide the spines on all four sides
# ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
#
# # Show the plot
# plt.show()
#In this version of the code, we plot the trace data against x-axis values that start at 0 but are shifted by 2380 points.




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


