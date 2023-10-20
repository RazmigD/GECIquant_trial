import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# filtered_trace = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/SPW_1.csv')
# spw_detection = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/SPWtime_1.csv')
#
# df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_32_37_slice1_1.csv')
# df_features = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_32_37_slice1_2.csv')
# #
# df = df.T  # shifts columns and rows
# indexes = []
# for index_row in df_features:
#     indexes.append(index_row)
#
# # replace column names (e.g. Event 3)  with only numbers (3)
# for i in range(len(df.columns)):
#     df.rename(columns={df.columns[i]: str(i + 1)}, inplace=True)
#
# # iterate over the column indexes and drop any columns that are not in the list
# for col_name in df.columns:
#     if col_name not in indexes:
#         df = df.drop(col_name, axis=1)
#
# # df2 = df.iloc[:, 175:185]
#
# df = df.drop(df.index[-1])
# col_order = list(df.iloc[1].argsort())
#
# # rearrange the columns of the dataframe based on the order of the second row
# df = df.iloc[:, col_order]

#print(df)

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


# Check calcium event onsets before and after sharp wave events

from load_data import load_data_
#df, df_features, df_mea, df_sw_events = load_data()
df, df_features, df_mea, df_sw_events = load_data_()

def process_spw(df_mea, df_sw_events):
    import pandas as pd
    import pickle
    df_summary = pd.read_csv('df_summary.csv')

    print(df_summary)


    calcium_onsets = (df.iloc[0, :]) / 3.09 # Frames to seconds
    if len(df_mea) > 97100:
        df_sw_events.iloc[:, [0, 1, 3]] = (df_sw_events.iloc[:, [0, 1, 3]] - 2380)

    sw_durations = df_sw_events.iloc[: ,2] / 1000  # SW event durations

    avg_sw_duration = sw_durations.mean()
    print('Average SW duration', avg_sw_duration)

    sharp_wave_onsets = (df_sw_events.iloc[:, 0])/1000 # ms to s
    sharp_wave_ends = (df_sw_events.iloc[:, 1])/1000 # ms to s

    spw_freq = len(df_sw_events)/100
    print("spw frequency is:", spw_freq)

    col_indexes = df.columns.values
    col_indexes = col_indexes.tolist()
    print('column indexes:', col_indexes)
    print(len(col_indexes))

    my_dict = {col_indexes[i]: calcium_onsets[i] for i in range(len(calcium_onsets))}
    print(my_dict)
    print(df_features.columns)
    #pdb.set_trace()
    print(len(df_features.columns))
    print(len(df_sw_events) / 100)


    with open('SPW_frequencies.pkl', 'rb') as f:
        SPW_frequencies = pickle.load(f)

    with open('ca_event_num.pkl', 'rb') as f:
        ca_event_num = pickle.load(f)

    SPW_frequencies.append(len(df_features.columns))
    ca_event_num.append(len(df_sw_events) / 100)

    with open('SPW_frequencies.pkl', 'wb') as f:
        pickle.dump(SPW_frequencies, f)

    with open('ca_event_num.pkl', 'wb') as f:
        pickle.dump(ca_event_num, f)

    print(SPW_frequencies)
    print(ca_event_num)


    events_before = []
    events_within = []
    events_after = []
    events_between = []

    # Load the summary dataframe from a csv file
    df_summary = pd.read_csv('df_summary.csv')

    with open('before_indexes.pkl', 'rb') as f:
        before_indexes = pickle.load(f)

    with open('within_indexes.pkl', 'rb') as f:
        within_indexes = pickle.load(f)

    with open('after_indexes.pkl', 'rb') as f:
        after_indexes = pickle.load(f)

    with open('between_indexes.pkl', 'rb') as f:
        between_indexes = pickle.load(f)

    for sharp_wave_onset in sharp_wave_onsets:
        calcium_onsets_b4_after = [x for x in my_dict.values() if
                                   x >= sharp_wave_onset - avg_sw_duration and x <= sharp_wave_onset + avg_sw_duration]
        print(f"sharp_wave_onset: {sharp_wave_onset}, calcium_onsets_b4_after: {calcium_onsets_b4_after}")

        for k, v in my_dict.items():
            if v in calcium_onsets_b4_after and v < sharp_wave_onset:
                print(f"Appending {k}: {v} to events_before")
                events_before.append({k: v})
            elif v in calcium_onsets_b4_after and v > sharp_wave_onset:
                print(f"Appending {k}: {v} to events_within")
                events_within.append({k: v})

    for sharp_wave_end in sharp_wave_ends:
        calcium_onsets_post_end = [x for x in my_dict.values() if
                                   x >= sharp_wave_end and x <= sharp_wave_end + avg_sw_duration]
        print(f"sharp_wave_end: {sharp_wave_end}, calcium_onsets_post_end: {calcium_onsets_post_end}")

        for k, v in my_dict.items():
            if v in calcium_onsets_post_end:
                print(f"Appending {k}: {v} to events_after")
                events_after.append({k: v})

    for i in range(len(sharp_wave_onsets) - 1):
        start = sharp_wave_ends[i] + avg_sw_duration
        end = sharp_wave_onsets[i + 1] - avg_sw_duration
        calcium_events_between = [x for x in my_dict.values() if start <= x <= end]

        for k, v in my_dict.items():
            if v in calcium_events_between:
                print(f"Appending {k}: {v} to events_between")
                events_between.append({k: v})

    print()
    print('Events before SW onset:', events_before)
    prop_before = (len(events_before) / len(my_dict)) * 100
    print('Proportion:', prop_before)

    print('Events within SW events:', events_within)
    prop_within = (len(events_within) / len(my_dict)) * 100
    print('Proportion:', prop_within)

    print('Events after SW end:', events_after)
    prop_after = (len(events_after) / len(my_dict)) * 100
    print('Proportion:', prop_after)

    print('Events between SW events:', events_between)
    prop_between = (len(events_between) / len(my_dict)) * 100
    print('Proportion:', prop_between)

    keys_before = [list(d.keys()) for d in events_before]
    keys_within = [list(d.keys()) for d in events_within]
    keys_after = [list(d.keys()) for d in events_after]
    keys_between = [list(d.keys()) for d in events_between]

    target_index = int(input("Enter the index of the target array: ")) - 1

    # Append the keys_only list to the target array
    before_indexes[target_index].append(keys_before)
    within_indexes[target_index].append(keys_within)
    after_indexes[target_index].append(keys_after)
    between_indexes[target_index].append(keys_between)

    with open('before_indexes.pkl', 'wb') as f:
        pickle.dump(before_indexes, f)

    with open('within_indexes.pkl', 'wb') as f:
        pickle.dump(within_indexes, f)

    with open('after_indexes.pkl', 'wb') as f:
        pickle.dump(after_indexes, f)

    with open('between_indexes.pkl', 'wb') as f:
        pickle.dump(between_indexes, f)

    print(events_before)
    print('Events before',before_indexes)
    print( )
    print(events_after)
    print('Events after',after_indexes)
    print( )
    print(events_within)
    print('Events within', within_indexes)
    print( )
    print(events_between)
    print('Events between', between_indexes)

    #
    # import pandas as pd
    #
    # # Ask the user for the index of the chosen subfolder
    # chosen_index = int(input("Enter the index of the chosen subfolder: "))
    #
    # chosen_index = int(chosen_index) - 1
    #
    # # Check if the chosen index is valid
    # if chosen_index not in df_summary.index:
    #     print("Invalid index.")
    # else:
    #     # Get the subfolder name corresponding to the chosen index
    #     subfolder_name = df_summary.at[chosen_index, 'subfolder_name']
    #
    #     # Modify the existing row in the DataFrame with the new mean time differences,
    #     # SPW freq, events before, and events after
    #     df_summary.at[chosen_index, 'SPW frequency'] = spw_freq
    #     df_summary.at[chosen_index, 'Average SW duration'] = avg_sw_duration
    #     df_summary.at[chosen_index, 'Events Before'] = prop_before
    #     df_summary.at[chosen_index, 'Events During'] = prop_within
    #     df_summary.at[chosen_index, 'Events After'] = prop_after
    #     df_summary.at[chosen_index, 'Events Uncorelated'] = prop_between
    #
    # #df_summary = df_summary.transpose()
    #
    #     #df_summary = df.drop(columns=['spw_freq'])
    #
    #     # Rename the columns to adjust the names
    #     #df_summary = df_summary.rename(columns={'SPW freq': 'SPW freq', 'events before': 'events before', 'events after': 'events after'})
    #
    #     # Save the updated summary dataframe
    #     df_summary.to_csv('df_summary.csv', index=False)
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.max_rows', None)
    #     pd.set_option('display.max_colwidth', None)
    #     pd.set_option('display.expand_frame_repr', False)
    #     print(df_summary)
    #














    # # Checking for calcium events within Sharp Waves
    #
    # # calcium_onsets = (df.iloc[1,:])/3
    # calcium_ends = (df.iloc[1, :]) / 3
    # calcium_peaks = (calcium_onsets + calcium_ends) / 2
    #
    # #print("Calcium Onsets", list(calcium_onsets))
    #
    # #print("Calcium Ends", list(calcium_ends))
    # print( )
    # #print("Calcium Peaks", list(calcium_peaks))
    # #2sharp_wave_onsets = (spw_detection.iloc[:, 0]) / 1000
    # sharp_wave_ends = (df_sw_events.iloc[:, 1]) / 1000
    # sharp_wave_peaks = (df_sw_events.iloc[:, 3]) / 1000

    # # Initialize counters for events before and after sharp waves
    # events = 0
    # peaks_before_sw = 0
    # peaks_after_sw = 0
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
    #
    #     for calcium_peak in calcium_peaks_within_spw:
    #         if calcium_peak < sharp_wave_peaks[i]:
    #             peaks_before_sw += 1
    #         elif calcium_peak > sharp_wave_peaks[i]:
    #             peaks_after_sw += 1
    #         else:
    #             events += 1
    #
    # # Print out counts of events before, within, and after sharp waves
    # print(f"{peaks_before_sw} calcium event peaks occurred before sharp wave peaks.")
    # #print(f"{events} calcium event peaks occurred at sharp wave peaks.")
    # print(f"{peaks_after_sw} calcium event peaks occurred after sharp wave peaks.")
    # print(f"No calcium event peak detected within sharp wave event window {no_peak} times.")



# # Histogram of Time Differences
#
#     # Create a histogram of the time differences
#     plt.hist(time_differences, bins=20)
#     plt.xlabel('Time difference (s)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of time differences between calcium and sharp wave events')
#     plt.show()

# # Plotting the SPW trace
#
#     x_values_green = df_sw_events.iloc[:, 0]
#     x_values_red = df_sw_events.iloc[:, 1]
#
#     # Plot the trace data
#     fig, ax = plt.subplots()
#
#     ax.plot(df_mea.iloc[:, 1])
#     ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
#     # Set the x-axis limits to zoom in on a specific region
#     # plt.xlim(25000, 35000)
#
#     # Draw the green and red vertical lines
#     for x in x_values_green:
#         plt.axvline(x=x, color='green', linewidth=0.3)
#
#     for x in x_values_red:
#         plt.axvline(x=x, color='red', linewidth=0.3)
#
#     # Show the plot
#     plt.title('Sharp Wave Ripple Events')
#     plt.xlabel('Time (ms)')
#     plt.show()











#
# # Plotting the average Sharp Wave Trace
#
#     # Initialize empty list to store sharp wave event traces
#     sharp_wave_traces = []
#
#     # Loop through each sharp wave event in spw_detection
#     for onset, end in zip(df_sw_events.iloc[:, 0], df_sw_events.iloc[:, 1]):
#     # Slice the filtered trace using onset and end points
#         sharp_wave_trace = df_mea.iloc[onset:end, 1].values
#     # Append the sharp wave event trace to the list
#         sharp_wave_traces.append(sharp_wave_trace)
#
#     # Determine the maximum length of all the sharp wave event traces
#     max_length = max([len(trace) for trace in sharp_wave_traces])
#
#     # Pad the shorter traces with zeros to ensure that they all have the same length
#     padded_traces = [np.pad(trace, (0, max_length - len(trace))) for trace in sharp_wave_traces]
#
#     # Compute mean of sharp wave event traces to obtain average sharp wave trace
#     average_sharp_wave_trace = sum(padded_traces) / len(padded_traces)
#
#     # Plot the average sharp wave trace
#     plt.plot(average_sharp_wave_trace)
#     plt.title('Average Sharp Wave Trace')
#     plt.show()




# ------------------------------------------------------------------------------------------------------

# #Plot Sharp Wave Trace

#Define the x-values for the vertical lines


# from data_loader import load_data
# _, _, df_mea, df_sw_events = load_data()
#
# def process_spw(df_mea, df_sw_events):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     x_values_green = df_sw_events.iloc[:, 0]
#     x_values_red = df_sw_events.iloc[:, 1]
#
#     # Plot the trace data
#     fig, ax = plt.subplots()
#
#     ax.plot(df_mea.iloc[:, 1])
#     ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
#     # Set the x-axis limits to zoom in on a specific region
#     #plt.xlim(25000, 35000)
#
#     # Draw the green and red vertical lines
#     for x in x_values_green:
#         plt.axvline(x=x, color='green', linewidth=0.3)
#
#     for x in x_values_red:
#         plt.axvline(x=x, color='red', linewidth=0.3)
#
#     # Show the plot
#     plt.show()

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


