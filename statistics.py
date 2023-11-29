import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

#
# df_mea = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0003_11_10_08_trace.csv')
# df_sw_events = spw_detection = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0004_11_10_08_SWevents.csv')
#
# df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0001_11_10_08_slice1.csv')
# df_features = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data/11_10_08_slice1/0002_11_10_08_slice2.csv')

from load_data import load_data_

df, df_features, df_mea, df_sw_events = load_data_()  # Replace 0 with the desired subfolder_choice


df = df.T  # shifts columns and rows
indexes = []
for index_row in df_features:
    indexes.append(index_row)

# Replace column names (e.g. Event 3)  with only numbers (3)
for i in range(len(df.columns)):
    df.rename(columns={df.columns[i]: str(i + 1)}, inplace=True)

# Iterate over the column indexes and drop any columns that are not in the list
for col_name in df.columns:
    if col_name not in indexes:
        df = df.drop(col_name, axis=1)

# df2 = df.iloc[:, 175:185]

df = df.drop(df.index[-1])
df = df.drop(df.index[0])

col_order = list(df.iloc[1].argsort())

df_features.to_csv('df_summary.csv', index=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
print(df_features)


df_summary_ = pd.read_csv('df_summary.csv')

##################### CONVERSION OF INDEXES from ['2'], ['3']... to [2], [3] #############################

with open('before_indexes.pkl', 'rb') as f:
    before_indexes = pickle.load(f)

with open('within_indexes.pkl', 'rb') as f:
    within_indexes = pickle.load(f)

with open('after_indexes.pkl', 'rb') as f:
    after_indexes = pickle.load(f)

with open('between_indexes.pkl', 'rb') as f:
    between_indexes = pickle.load(f)

print("Indexes of events before SW onsets", before_indexes)
print("Indexes of events after SW onsets", after_indexes)
print("Indexes of events within SW onsets", within_indexes)


print()

before_indexes_ = []
after_indexes_ = []
within_indexes_ = []
between_indexes_ = []

#
[before_indexes_.extend(sublist) for sublist in before_indexes]
before_indexes_ = [[[int(item) for sublist in inner_list for item in sublist] for inner_list in sublist] for sublist in before_indexes]
print("Before indexes", before_indexes_)


[within_indexes_.extend(sublist) for sublist in within_indexes]
within_indexes_ = [[[int(item) for sublist in inner_list for item in sublist] for inner_list in sublist] for sublist in within_indexes]
print("Within indexes", within_indexes_)


[after_indexes_.extend(sublist) for sublist in after_indexes]
after_indexes_ = [[[int(item) for sublist in inner_list for item in sublist] for inner_list in sublist] for sublist in after_indexes]
print("After indexes", after_indexes_)

[between_indexes_.extend(sublist) for sublist in between_indexes]
between_indexes_ = [[[int(item) for sublist in inner_list for item in sublist] for inner_list in sublist] for sublist in between_indexes]
print("Between indexes", between_indexes_)
#

#pdb.set_trace()

##################################################################################################################

with open('before_areas.pkl', 'rb') as f:
    before_areas = pickle.load(f)
with open('before_dur.pkl', 'rb') as f:
    before_dur = pickle.load(f)
with open('before_dff.pkl', 'rb') as f:
    before_dff = pickle.load(f)
with open('before_rise.pkl', 'rb') as f:
    before_rise = pickle.load(f)
with open('before_decay.pkl', 'rb') as f:
    before_decay = pickle.load(f)
with open('before_propagation.pkl', 'rb') as f:
    before_propagation = pickle.load(f)
with open('before_sp_dens.pkl', 'rb') as f:
    before_sp_dens = pickle.load(f)
with open('before_temp_dens.pkl', 'rb') as f:
    before_temp_dens = pickle.load(f)
with open('before_dur_10.pkl', 'rb') as f:
    before_dur_10 = pickle.load(f)



with open('within_areas.pkl', 'rb') as f:
    within_areas = pickle.load(f)
with open('within_dur.pkl', 'rb') as f:
    within_dur = pickle.load(f)
with open('within_dff.pkl', 'rb') as f:
    within_dff = pickle.load(f)
with open('within_rise.pkl', 'rb') as f:
    within_rise = pickle.load(f)
with open('within_decay.pkl', 'rb') as f:
    within_decay = pickle.load(f)
with open('within_propagation.pkl', 'rb') as f:
    within_propagation = pickle.load(f)
with open('within_sp_dens.pkl', 'rb') as f:
    within_sp_dens = pickle.load(f)
with open('within_temp_dens.pkl', 'rb') as f:
    within_temp_dens = pickle.load(f)
with open('within_dur_10.pkl', 'rb') as f:
    within_dur_10 = pickle.load(f)


with open('after_areas.pkl', 'rb') as f:
    after_areas = pickle.load(f)
with open('after_dur.pkl', 'rb') as f:
    after_dur = pickle.load(f)
with open('after_dff.pkl', 'rb') as f:
    after_dff = pickle.load(f)
with open('after_rise.pkl', 'rb') as f:
    after_rise = pickle.load(f)
with open('after_decay.pkl', 'rb') as f:
    after_decay = pickle.load(f)
with open('after_propagation.pkl', 'rb') as f:
    after_propagation = pickle.load(f)
with open('after_sp_dens.pkl', 'rb') as f:
    after_sp_dens = pickle.load(f)
with open('after_temp_dens.pkl', 'rb') as f:
    after_temp_dens = pickle.load(f)
with open('after_dur_10.pkl', 'rb') as f:
    after_dur_10 = pickle.load(f)


with open('between_areas.pkl', 'rb') as f:
    between_areas = pickle.load(f)
with open('between_dur.pkl', 'rb') as f:
    between_dur = pickle.load(f)
with open('between_dff.pkl', 'rb') as f:
    between_dff = pickle.load(f)
with open('between_rise.pkl', 'rb') as f:
    between_rise = pickle.load(f)
with open('between_decay.pkl', 'rb') as f:
    between_decay = pickle.load(f)
with open('between_propagation.pkl', 'rb') as f:
    between_propagation = pickle.load(f)
with open('between_sp_dens.pkl', 'rb') as f:
    between_sp_dens = pickle.load(f)
with open('between_temp_dens.pkl', 'rb') as f:
    between_temp_dens = pickle.load(f)
with open('between_dur_10.pkl', 'rb') as f:
    between_dur_10 = pickle.load(f)

######################################### GETTING THE AREA/DUR/DFF/RISE/DECAY VARIABLES ###################################

# #
target_index = int(input("Enter the index of the target array: ")) - 1
print(target_index)

# Hold the combined elements from the before_indexes_[target_index], after_indexes_[target_index]), ...
  # Combines the sublists of indexes [[2],[3],[5],...], [[]], [[]]] of the target index and puts them in a list [2,3,4,...]

before_indices = sum(list(before_indexes_[target_index]), [])
after_indices = sum(list(after_indexes_[target_index]), [])
within_indices = sum(list(within_indexes_[target_index]), [])
between_indices = sum(list(between_indexes_[target_index]), [])

# For every value in df_features if a match exists in before_indices, after_indices...,
  # appends specific features to corresponding lists

for col_name in df_features.columns:
    try:
        col_idx = int(col_name)

        if col_idx in before_indices:
            before_areas[target_index].append(df_features.loc[0, col_name])
            before_dur[target_index].append(df_features.loc[5, col_name])
            before_dff[target_index].append(df_features.loc[4, col_name])
            before_rise[target_index].append(df_features.loc[7, col_name])
            before_decay[target_index].append(df_features.loc[8, col_name])
            before_propagation_value = df_features.loc[10, col_name]
            if before_propagation_value != 0:
                before_propagation[target_index].append(before_propagation_value)
            before_sp_dens[target_index].append(df_features.loc[32, col_name])
            before_temp_dens[target_index].append(df_features.loc[30, col_name])
            before_dur_10[target_index].append(df_features.loc[6, col_name])

        if col_idx in after_indices:
            after_areas[target_index].append(df_features.loc[0, col_name])
            after_dur[target_index].append(df_features.loc[5, col_name])
            after_dff[target_index].append(df_features.loc[4, col_name])
            after_rise[target_index].append(df_features.loc[7, col_name])
            after_decay[target_index].append(df_features.loc[8, col_name])
            after_propagation_value = df_features.loc[10, col_name]
            if after_propagation_value != 0:
                after_propagation[target_index].append(after_propagation_value)
            after_sp_dens[target_index].append(df_features.loc[32, col_name])
            after_temp_dens[target_index].append(df_features.loc[30, col_name])
            after_dur_10[target_index].append(df_features.loc[6, col_name])

        if col_idx in within_indices:
            within_areas[target_index].append(df_features.loc[0, col_name])
            within_dur[target_index].append(df_features.loc[5, col_name])
            within_dff[target_index].append(df_features.loc[4, col_name])
            within_rise[target_index].append(df_features.loc[7, col_name])
            within_decay[target_index].append(df_features.loc[8, col_name])
            within_propagation_value = df_features.loc[10, col_name]
            if within_propagation_value != 0:
                within_propagation[target_index].append(within_propagation_value)
            within_sp_dens[target_index].append(df_features.loc[32, col_name])
            within_temp_dens[target_index].append(df_features.loc[30, col_name])
            within_dur_10[target_index].append(df_features.loc[6, col_name])

        if col_idx in between_indices:
            between_areas[target_index].append(df_features.loc[0, col_name])
            between_dur[target_index].append(df_features.loc[5, col_name])
            between_dff[target_index].append(df_features.loc[4, col_name])
            between_rise[target_index].append(df_features.loc[7, col_name])
            between_decay[target_index].append(df_features.loc[8, col_name])
            between_propagation_value = df_features.loc[10, col_name]
            if between_propagation_value != 0:
                between_propagation[target_index].append(between_propagation_value)
            between_sp_dens[target_index].append(df_features.loc[32, col_name])
            between_temp_dens[target_index].append(df_features.loc[30, col_name])
            between_dur_10[target_index].append(df_features.loc[6, col_name])
    except ValueError:
        pass

# #
print(f"Areas for calcium events BEFORE SW events{before_areas}")
print(f"Areas for calcium events AFTER SW events{after_areas}")
print(f"Areas for calcium events WITHIN SW events{within_areas}")
print(f"Areas for calcium events BETWEEN SW events{between_areas}")

print( )
print(f"50-50 dur for calcium events BEFORE SW events{before_dur}")
print(f"50-50 dur for calcium events AFTER SW events{after_dur}")
print(f"50-50 dur calcium events WITHIN SW events{within_dur}")
print(f"50-50 dur calcium events BETWEEN SW events{between_dur}")


print(f"Dff for calcium events BEFORE SW events{before_dff}")
print(f"Dff for calcium events AFTER SW events{after_dff}")
print(f"Dff for calcium events WITHIN SW events{within_dff}")
print(f"Dff for calcium events BETWEEN SW events{between_dff}")
#
#
print(f"Rise time for calcium events BEFORE SW events{before_rise}")
print(f"Rise time for calcium events AFTER SW events{after_rise}")
print(f"Rise time for calcium events WITHIN SW events{within_rise}")
print(f"Rise time for calcium events BETWEEN SW events{between_rise}")


print(f"Propagation for calcium events BEFORE SW events{before_propagation}")
print(f"Propagation time for calcium events AFTER SW events{after_propagation}")
print(f"Propagation time for calcium events WITHIN SW events{within_propagation}")
print(f"Propagation time for calcium events BETWEEN SW events{between_propagation}")


with open('before_areas.pkl', 'wb') as f:
    pickle.dump(before_areas, f)

with open('within_areas.pkl', 'wb') as f:
    pickle.dump(within_areas, f)

with open('after_areas.pkl', 'wb') as f:
    pickle.dump(after_areas, f)

with open('between_areas.pkl', 'wb') as f:
    pickle.dump(between_areas, f)

#################################


with open('before_dur.pkl', 'wb') as f:
    pickle.dump(before_dur, f)

with open('within_dur.pkl', 'wb') as f:
    pickle.dump(within_dur, f)

with open('after_dur.pkl', 'wb') as f:
    pickle.dump(after_dur, f)

with open('between_dur.pkl', 'wb') as f:
    pickle.dump(between_dur, f)

################################

with open('before_dff.pkl', 'wb') as f:
    pickle.dump(before_dff, f)

with open('within_dff.pkl', 'wb') as f:
    pickle.dump(within_dff, f)

with open('after_dff.pkl', 'wb') as f:
    pickle.dump(after_dff, f)

with open('between_dff.pkl', 'wb') as f:
    pickle.dump(between_dff, f)

#############################

with open('before_rise.pkl', 'wb') as f:
    pickle.dump(before_rise, f)

with open('within_rise.pkl', 'wb') as f:
    pickle.dump(within_rise, f)

with open('after_rise.pkl', 'wb') as f:
    pickle.dump(after_rise, f)

with open('between_rise.pkl', 'wb') as f:
    pickle.dump(between_rise, f)

#############################

with open('before_decay.pkl', 'wb') as f:
    pickle.dump(before_decay, f)

with open('within_decay.pkl', 'wb') as f:
    pickle.dump(within_decay, f)

with open('after_decay.pkl', 'wb') as f:
    pickle.dump(after_decay, f)

with open('between_decay.pkl', 'wb') as f:
    pickle.dump(between_decay, f)

####################################
with open('before_propagation.pkl', 'wb') as f:
    pickle.dump(before_propagation, f)

with open('within_propagation.pkl', 'wb') as f:
    pickle.dump(within_propagation, f)

with open('after_propagation.pkl', 'wb') as f:
    pickle.dump(after_propagation, f)

with open('between_propagation.pkl', 'wb') as f:
    pickle.dump(between_propagation, f)

########################################

with open('before_sp_dens.pkl', 'wb') as f:
    pickle.dump(before_sp_dens, f)

with open('after_sp_dens.pkl', 'wb') as f:
    pickle.dump(after_sp_dens, f)

with open('within_sp_dens.pkl', 'wb') as f:
    pickle.dump(within_sp_dens, f)

with open('between_sp_dens.pkl', 'wb') as f:
    pickle.dump(between_sp_dens, f)

######################################

with open('before_temp_dens.pkl', 'wb') as f:
    pickle.dump(before_temp_dens, f)

with open('after_temp_dens.pkl', 'wb') as f:
    pickle.dump(after_temp_dens, f)

with open('within_temp_dens.pkl', 'wb') as f:
    pickle.dump(within_temp_dens, f)

with open('between_temp_dens.pkl', 'wb') as f:
    pickle.dump(between_temp_dens, f)


#######################################

with open('before_dur_10.pkl', 'wb') as f:
    pickle.dump(before_dur_10, f)

with open('after_dur_10.pkl', 'wb') as f:
    pickle.dump(after_dur_10, f)

with open('within_dur_10.pkl', 'wb') as f:
    pickle.dump(within_dur_10, f)

with open('between_dur_10.pkl', 'wb') as f:
    pickle.dump(between_dur_10, f)



############################################################################################################

#
#################################### BOXPLOTS FOR AREA ND DURATION ########################################
#
# import itertools
# import matplotlib.pyplot as plt
# #
# # Flatten the lists
# before_areas_flat = list(itertools.chain(*before_areas))
# before_areas_flat = np.array(before_areas_flat)
# after_areas_flat = list(itertools.chain(*after_areas))
# after_areas_flat = np.array(after_areas_flat)
# within_areas_flat = list(itertools.chain(*within_areas))
# within_areas_flat = np.array(within_areas_flat)
# between_areas_flat = list(itertools.chain(*between_areas))
# between_areas_flat = np.array(between_areas_flat)
#
# before_dur_flat = list(itertools.chain(*before_dur))
# before_dur_flat = np.array(before_dur_flat)
# after_dur_flat = list(itertools.chain(*after_dur))
# after_dur_flat = np.array(after_dur_flat)
# within_dur_flat = list(itertools.chain(*within_dur))
# within_dur_flat = np.array(within_dur_flat)
# between_dur_flat = list(itertools.chain(*between_dur))
# between_dur_flat = np.array(between_dur_flat)
#
# before_dff_flat = list(itertools.chain(*before_dff))
# before_dff_flat = np.array(before_dff_flat)
# after_dff_flat = list(itertools.chain(*after_dff))
# after_dff_flat = np.array(after_dff_flat)
# within_dff_flat = list(itertools.chain(*within_dff))
# within_dff_flat = np.array(within_dff_flat)
# between_dff_flat = list(itertools.chain(*between_dff))
# between_dff_flat = np.array(between_dff_flat)
#
# before_rise_flat = list(itertools.chain(*before_rise))
# before_rise_flat = np.array(before_rise_flat)
# after_rise_flat = list(itertools.chain(*after_rise))
# after_rise_flat = np.array(after_rise_flat)
# within_rise_flat = list(itertools.chain(*within_rise))
# within_rise_flat = np.array(within_rise_flat)
# between_rise_flat = list(itertools.chain(*between_rise))
# between_rise_flat = np.array(between_rise_flat)
#
# before_decay_flat = list(itertools.chain(*before_decay))
# before_decay_flat = np.array(before_decay_flat)
# after_decay_flat = list(itertools.chain(*after_decay))
# after_decay_flat = np.array(after_decay_flat)
# within_decay_flat = list(itertools.chain(*within_decay))
# within_decay_flat = np.array(within_decay_flat)
# between_decay_flat = list(itertools.chain(*between_decay))
# between_decay_flat = np.array(between_decay_flat)
#
# before_propagation_flat = list(itertools.chain(*before_propagation))
# before_propagation_flat = np.array(before_propagation_flat)
# after_propagation_flat = list(itertools.chain(*after_propagation))
# after_propagation_flat = np.array(after_propagation_flat)
# within_propagation_flat = list(itertools.chain(*within_propagation))
# within_propagation_flat = np.array(within_propagation_flat)
# between_propagation_flat = list(itertools.chain(*between_propagation))
# between_propagation_flat = np.array(between_propagation_flat)
#
# before_sp_dens_flat = list(itertools.chain(*before_sp_dens))
# before_sp_dens_flat = np.array(before_sp_dens_flat)
# after_sp_dens_flat = list(itertools.chain(*after_sp_dens))
# after_sp_dens_flat = np.array(after_sp_dens_flat)
# within_sp_dens_flat = list(itertools.chain(*within_sp_dens))
# within_sp_dens_flat = np.array(within_sp_dens_flat)
# between_sp_dens_flat = list(itertools.chain(*between_sp_dens))
# between_sp_dens_flat = np.array(between_sp_dens_flat)
#
# before_temp_dens_flat = list(itertools.chain(*before_temp_dens))
# before_temp_dens_flat = np.array(before_temp_dens_flat)
# after_temp_dens_flat = list(itertools.chain(*after_temp_dens))
# after_temp_dens_flat = np.array(after_temp_dens_flat)
# within_temp_dens_flat = list(itertools.chain(*within_temp_dens))
# within_temp_dens_flat = np.array(within_temp_dens_flat)
# between_temp_dens_flat = list(itertools.chain(*between_temp_dens))
# between_temp_dens_flat = np.array(between_temp_dens_flat)
#
# before_dur_10_flat = list(itertools.chain(*before_dur_10))
# before_dur_10_flat = np.array(before_dur_10_flat)
# after_dur_10_flat = list(itertools.chain(*after_dur_10))
# after_dur_10_flat = np.array(after_dur_10_flat)
# within_dur_10_flat = list(itertools.chain(*within_dur_10))
# within_dur_10_flat = np.array(within_dur_10_flat)
# between_dur_10_flat = list(itertools.chain(*between_dur_10))
# between_dur_10_flat = np.array(between_dur_10_flat)
#
# # Calculate the quartiles and interquartile range all variables
#
# def remove_outliers(data):
#     q1, q3 = np.percentile(data, [25, 75])
#     iqr = q3 - q1
#
#     # Define the outlier threshold as 1.5 times the IQR
#     lower_threshold = q1 - (1.5 * iqr)
#     upper_threshold = q3 + (1.5 * iqr)
#
#     # Remove the outliers
#     cleaned_data = data[(data >= lower_threshold) & (data <= upper_threshold)]
#     return cleaned_data
#
# # Remove outliers for each group
# cleaned_before_areas = remove_outliers(before_areas_flat)
# cleaned_after_areas = remove_outliers(after_areas_flat)
# cleaned_within_areas = remove_outliers(within_areas_flat)
# cleaned_between_areas = remove_outliers(between_areas_flat)
#
# cleaned_before_dur = remove_outliers(before_dur_flat)
# cleaned_after_dur = remove_outliers(after_dur_flat)
# cleaned_within_dur = remove_outliers(within_dur_flat)
# cleaned_between_dur = remove_outliers(between_dur_flat)
#
# cleaned_before_dff = remove_outliers(before_dff_flat)
# cleaned_after_dff = remove_outliers(after_dff_flat)
# cleaned_within_dff = remove_outliers(within_dff_flat)
# cleaned_between_dff = remove_outliers(between_dff_flat)
#
# cleaned_before_rise = remove_outliers(before_rise_flat)
# cleaned_after_rise = remove_outliers(after_rise_flat)
# cleaned_within_rise = remove_outliers(within_rise_flat)
# cleaned_between_rise = remove_outliers(between_rise_flat)
#
# cleaned_before_decay = remove_outliers(before_decay_flat)
# cleaned_after_decay = remove_outliers(after_decay_flat)
# cleaned_within_decay = remove_outliers(within_decay_flat)
# cleaned_between_decay = remove_outliers(between_decay_flat)
#
# cleaned_before_propagation = remove_outliers(before_propagation_flat)
# cleaned_after_propagation = remove_outliers(after_propagation_flat)
# cleaned_within_propagation = remove_outliers(within_propagation_flat)
# cleaned_between_propagation = remove_outliers(between_propagation_flat)
#
# cleaned_before_sp_dens = remove_outliers(before_sp_dens_flat)
# cleaned_after_sp_dens = remove_outliers(after_sp_dens_flat)
# cleaned_within_sp_dens = remove_outliers(within_sp_dens_flat)
# cleaned_between_sp_dens = remove_outliers(between_sp_dens_flat)
#
# cleaned_before_temp_dens= remove_outliers(before_temp_dens_flat)
# cleaned_after_temp_dens = remove_outliers(after_temp_dens_flat)
# cleaned_within_temp_dens = remove_outliers(within_temp_dens_flat)
# cleaned_between_temp_dens = remove_outliers(between_temp_dens_flat)
#
# cleaned_before_dur_10 = remove_outliers(before_dur_10_flat)
# cleaned_after_dur_10 = remove_outliers(after_dur_10_flat)
# cleaned_within_dur_10 = remove_outliers(within_dur_10_flat)
# cleaned_between_dur_10 = remove_outliers(between_dur_10_flat)
#
#
# # #------------------CHECKING FOR NORMALITY----------------------------------------------
# #
#
# from scipy.stats import shapiro, f_oneway, kruskal
#
# # Perform Shapiro-Wilk test for normality
# _, p_value_before_areas = shapiro(before_areas_flat)
# _, p_value_after_areas = shapiro(after_areas_flat)
# _, p_value_within_areas = shapiro(within_areas_flat)
# _, p_value_between_areas = shapiro(between_areas_flat)
#
# _, p_value_before_dff = shapiro(cleaned_before_dff)
# _, p_value_after_dff = shapiro(cleaned_after_dff)
# _, p_value_within_dff = shapiro(cleaned_within_dff)
# _, p_value_between_dff = shapiro(cleaned_between_dff)
#
# _, p_value_before_dur = shapiro(cleaned_before_dur)
# _, p_value_after_dur = shapiro(cleaned_after_dur)
# _, p_value_within_dur = shapiro(cleaned_within_dur)
# _, p_value_between_dur = shapiro(cleaned_between_dur)
#
# _, p_value_before_rise = shapiro(cleaned_before_rise)
# _, p_value_after_rise = shapiro(cleaned_after_rise)
# _, p_value_within_rise = shapiro(cleaned_within_rise)
# _, p_value_between_rise = shapiro(cleaned_between_rise)
#
# _, p_value_before_decay = shapiro(cleaned_before_decay)
# _, p_value_after_decay = shapiro(cleaned_after_decay)
# _, p_value_within_decay = shapiro(cleaned_within_decay)
# _, p_value_between_decay = shapiro(cleaned_between_decay)
#
# _, p_value_before_propagation = shapiro(cleaned_before_propagation)
# _, p_value_after_propagation = shapiro(cleaned_after_propagation)
# _, p_value_within_propagation = shapiro(cleaned_within_propagation)
# _, p_value_between_propagation = shapiro(cleaned_between_propagation)
#
# _, p_value_before_sp_dens = shapiro(cleaned_before_sp_dens)
# _, p_value_after_sp_dens = shapiro(cleaned_after_sp_dens)
# _, p_value_within_sp_dens = shapiro(cleaned_within_sp_dens)
# _, p_value_between_sp_dens = shapiro(cleaned_between_sp_dens)
#
# _, p_value_before_temp_dens = shapiro(cleaned_before_temp_dens)
# _, p_value_after_temp_dens = shapiro(cleaned_after_temp_dens)
# _, p_value_within_temp_dens = shapiro(cleaned_within_temp_dens)
# _, p_value_between_temp_dens = shapiro(cleaned_between_temp_dens)
#
# _, p_value_before_dur_10 = shapiro(cleaned_before_dur_10)
# _, p_value_after_dur_10 = shapiro(cleaned_after_dur_10)
# _, p_value_within_dur_10 = shapiro(cleaned_within_dur_10)
# _, p_value_between_dur_10 = shapiro(cleaned_between_dur_10)
#
# # Check normality assumption for each group
# alpha = 0.05  # significance level
#
# # Check normality assumption for each group and display the results
# print("Areas:")
# print("Before Areas:", "Normal" if p_value_before_areas > alpha else "Not Normal")
# print("After Areas:", "Normal" if p_value_after_areas > alpha else "Not Normal")
# print("Within Areas:", "Normal" if p_value_within_areas > alpha else "Not Normal")
# print("Between Areas:", "Normal" if p_value_between_areas > alpha else "Not Normal")
#
# print("Dff:")
# print("Before Dff:", "Normal" if p_value_before_dff > alpha else "Not Normal")
# print("After Dff:", "Normal" if p_value_after_dff > alpha else "Not Normal")
# print("Within Dff:", "Normal" if p_value_within_dff > alpha else "Not Normal")
# print("Between Dff:", "Normal" if p_value_between_dff > alpha else "Not Normal")
#
# print("Dur:")
# print("Before Dur:", "Normal" if p_value_before_dur > alpha else "Not Normal")
# print("After Dur:", "Normal" if p_value_after_dur > alpha else "Not Normal")
# print("Within Dur:", "Normal" if p_value_within_dur > alpha else "Not Normal")
# print("Between Dur:", "Normal" if p_value_between_dur > alpha else "Not Normal")
#
# print("Rise:")
# print("Before Rise:", "Normal" if p_value_before_rise > alpha else "Not Normal")
# print("After Rise:", "Normal" if p_value_after_rise > alpha else "Not Normal")
# print("Within Rise:", "Normal" if p_value_within_rise > alpha else "Not Normal")
# print("Between Rise:", "Normal" if p_value_between_rise > alpha else "Not Normal")
#
# print("Decay:")
# print("Before Decay:", "Normal" if p_value_before_decay > alpha else "Not Normal")
# print("After Decay:", "Normal" if p_value_after_decay > alpha else "Not Normal")
# print("Within Decay:", "Normal" if p_value_within_decay > alpha else "Not Normal")
# print("Between Decay:", "Normal" if p_value_between_decay > alpha else "Not Normal")
#
# print("Propagation:")
# print("Before Propagation:", "Normal" if p_value_before_propagation > alpha else "Not Normal")
# print("After Propagation:", "Normal" if p_value_after_propagation > alpha else "Not Normal")
# print("Within Propagation:", "Normal" if p_value_within_propagation > alpha else "Not Normal")
# print("Between Propagation:", "Normal" if p_value_between_propagation > alpha else "Not Normal")
#
# print("Spatial Density:")
# print("Before SD:", "Normal" if p_value_before_sp_dens > alpha else "Not Normal")
# print("After SD:", "Normal" if p_value_after_sp_dens > alpha else "Not Normal")
# print("Within SD:", "Normal" if p_value_within_sp_dens > alpha else "Not Normal")
# print("Between SD:", "Normal" if p_value_between_sp_dens > alpha else "Not Normal")
#
# print("Temporal Density:")
# print("Before TD:", "Normal" if p_value_before_temp_dens > alpha else "Not Normal")
# print("After TD:", "Normal" if p_value_after_temp_dens > alpha else "Not Normal")
# print("Within TD:", "Normal" if p_value_within_temp_dens > alpha else "Not Normal")
# print("Between TD:", "Normal" if p_value_between_temp_dens > alpha else "Not Normal")
#
# print("Duration 10 - 10:")
# print("Before 10-10 Duration:", "Normal" if p_value_before_dur_10 > alpha else "Not Normal")
# print("After 10-10 Duration:", "Normal" if p_value_after_dur_10 > alpha else "Not Normal")
# print("Within 10-10 Duration:", "Normal" if p_value_within_dur_10 > alpha else "Not Normal")
# print("Between 10-10 Duration:", "Normal" if p_value_between_dur_10 > alpha else "Not Normal")
#
#
# #################### PERFORMING KRUSKAL WALLIS SINCE DATA NOT NORMAL ##############################
#
# from scipy.stats import kruskal
#
# import numpy as np
# from scipy.stats import kruskal
# import matplotlib.pyplot as plt
#
# # Perform Kruskal-Wallis test
# h_statistic_areas, p_value_areas = kruskal(cleaned_before_areas, cleaned_after_areas,
# cleaned_within_areas, cleaned_between_areas)
# h_statistic_dff, p_value_dff = kruskal(cleaned_before_dff, cleaned_after_dff,
# cleaned_within_dff, cleaned_between_dff)
# h_statistic_dur, p_value_dur = kruskal(cleaned_before_dur, cleaned_after_dur,
# cleaned_within_dur, cleaned_between_dur)
# h_statistic_rise, p_value_rise = kruskal(cleaned_before_rise, cleaned_after_rise,
# cleaned_within_rise, cleaned_between_rise)
# h_statistic_decay, p_value_decay = kruskal(cleaned_before_decay, cleaned_after_decay,
# cleaned_within_decay, cleaned_between_decay)
# h_statistic_propagation, p_value_propagation = kruskal(cleaned_before_propagation, cleaned_after_propagation,
# cleaned_within_propagation, cleaned_between_propagation)
# h_statistic_sp_dens, p_value_sp_dens = kruskal(cleaned_before_sp_dens, cleaned_after_sp_dens,
# cleaned_within_sp_dens)
# h_statistic_temp_dens, p_value_temp_dens = kruskal(cleaned_before_temp_dens, cleaned_after_temp_dens,
# cleaned_within_temp_dens)
# h_statistic_dur_10, p_value_dur_10 = kruskal(cleaned_before_dur_10, cleaned_after_dur_10,
# cleaned_within_dur_10, cleaned_between_dur_10)
#
# # Print the results
# print("Kruskal-Wallis Test Results:")
# print("Areas - H-statistic:", h_statistic_areas, "p-value:", p_value_areas)
# print("Dff - H-statistic:", h_statistic_dff, "p-value:", p_value_dff)
# print("Dur - H-statistic:", h_statistic_dur, "p-value:", p_value_dur)
# print("Rise - H-statistic:", h_statistic_rise, "p-value:", p_value_rise)
# print("Decay - H-statistic:", h_statistic_decay, "p-value:", p_value_decay)
# print("Propagation - H-statistic:", h_statistic_propagation, "p-value:", p_value_propagation)
# print("Sp_dens - H-statistic:", h_statistic_sp_dens, "p-value:", p_value_sp_dens)
# print("Temp_dens - H-statistic:", h_statistic_temp_dens, "p-value:", p_value_temp_dens)
# print("Dur_10 - H-statistic:", h_statistic_dur_10, "p-value:", p_value_dur_10)
# #
# # Function to calculate mean and standard error
# def calculate_mean_std_err(data):
#     mean = np.mean(data)
#     std_err = np.std(data) / np.sqrt(len(data))
#     return mean, std_err
#
# # Box plots for areas
# plt.boxplot([cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Area (μm**2)')
# plt.title('Area differences among Ca2+ event groups')
# plt.show()
#
# # Calculate and print mean +/- standard error for areas
# mean_before_areas, std_err_before_areas = calculate_mean_std_err(cleaned_before_areas)
# mean_after_areas, std_err_after_areas = calculate_mean_std_err(cleaned_after_areas)
# mean_within_areas, std_err_within_areas = calculate_mean_std_err(cleaned_within_areas)
# mean_between_areas, std_err_between_areas = calculate_mean_std_err(cleaned_between_areas)
#
# print("Areas:")
# print(f"Before: {mean_before_areas:.2f} +/- {std_err_before_areas:.2f}")
# print(f"After: {mean_after_areas:.2f} +/- {std_err_after_areas:.2f}")
# print(f"Within: {mean_within_areas:.2f} +/- {std_err_within_areas:.2f}")
# print(f"Between: {mean_between_areas:.2f} +/- {std_err_between_areas:.2f}")
#
# # Box plots for durations
# plt.boxplot([cleaned_before_dur, cleaned_after_dur, cleaned_within_dur, cleaned_between_dur])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Duration (s)')
# plt.title('50% to 50% duration differences among Ca2+ event groups')
# plt.show()
#
# # Calculate and print mean +/- standard error for durations
# mean_before_dur, std_err_before_dur = calculate_mean_std_err(cleaned_before_dur)
# mean_after_dur, std_err_after_dur = calculate_mean_std_err(cleaned_after_dur)
# mean_within_dur, std_err_within_dur = calculate_mean_std_err(cleaned_within_dur)
# mean_between_dur, std_err_between_dur = calculate_mean_std_err(cleaned_between_dur)
#
# print("Durations:")
# print(f"Before: {mean_before_dur:.2f} +/- {std_err_before_dur:.2f}")
# print(f"After: {mean_after_dur:.2f} +/- {std_err_after_dur:.2f}")
# print(f"Within: {mean_within_dur:.2f} +/- {std_err_within_dur:.2f}")
# print(f"Between: {mean_between_dur:.2f} +/- {std_err_between_dur:.2f}")
#
# # Box plots for dff
# plt.boxplot([cleaned_before_dff, cleaned_before_dff, cleaned_within_dff, cleaned_between_dff])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('ΔF/F differences among Ca2+ event groups')
# plt.ylabel('ΔF/F')
# #
# plt.show()
#
# # Calculate and print mean +/- standard error for dff
# mean_before_dff, std_err_before_dff = calculate_mean_std_err(cleaned_before_dff)
# mean_after_dff, std_err_after_dff = calculate_mean_std_err(cleaned_before_dff)
# mean_within_dff, std_err_within_dff = calculate_mean_std_err(cleaned_within_dff)
# mean_between_dff, std_err_between_dff = calculate_mean_std_err(cleaned_between_dff)
#
# print("Dff:")
# print(f"Before: {mean_before_dff:.2f} +/- {std_err_before_dff:.2f}")
# print(f"After: {mean_after_dff:.2f} +/- {std_err_after_dff:.2f}")
# print(f"Within: {mean_within_dff:.2f} +/- {std_err_within_dff:.2f}")
# print(f"Between: {mean_between_dff:.2f} +/- {std_err_between_dff:.2f}")
#
# # Box plots for rise time
# plt.boxplot([cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Rise time differences among Ca2+ groups')
# plt.ylabel('Rise time (s)')
# #
# plt.show()
#
# # Box plots for propagation
# plt.boxplot([cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Propagation differences among Ca2+ event groups')
# plt.ylabel('Propagation')
# plt.show()
#
# # Calculate and print mean +/- standard error for propagation
# mean_before_propagation, std_err_before_propagation = calculate_mean_std_err(cleaned_before_propagation)
# mean_after_propagation, std_err_after_propagation = calculate_mean_std_err(cleaned_after_propagation)
# mean_within_propagation, std_err_within_propagation = calculate_mean_std_err(cleaned_within_propagation)
# mean_between_propagation, std_err_between_propagation = calculate_mean_std_err(cleaned_between_propagation)
#
# print("Propagation:")
# print(f"Before: {mean_before_propagation:.2f} +/- {std_err_before_propagation:.2f}")
# print(f"After: {mean_after_propagation:.2f} +/- {std_err_after_propagation:.2f}")
# print(f"Within: {mean_within_propagation:.2f} +/- {std_err_within_propagation:.2f}")
# print(f"Between: {mean_between_propagation:.2f} +/- {std_err_between_propagation:.2f}")
#
# # Box plots for sp_dens
# plt.boxplot([cleaned_before_sp_dens, cleaned_after_sp_dens, cleaned_within_sp_dens, cleaned_between_sp_dens])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Spatial density differences among Ca2+ event groups')
# plt.ylabel('Spatial Density')
# plt.show()
#
# # Calculate and print mean +/- standard error for sp_dens
# mean_before_sp_dens, std_err_before_sp_dens = calculate_mean_std_err(cleaned_before_sp_dens)
# mean_after_sp_dens, std_err_after_sp_dens = calculate_mean_std_err(cleaned_after_sp_dens)
# mean_within_sp_dens, std_err_within_sp_dens = calculate_mean_std_err(cleaned_within_sp_dens)
# mean_between_sp_dens, std_err_between_sp_dens = calculate_mean_std_err(cleaned_between_sp_dens)
#
# print("Spatial Density:")
# print(f"Before: {mean_before_sp_dens:.2f} +/- {std_err_before_sp_dens:.2f}")
# print(f"After: {mean_after_sp_dens:.2f} +/- {std_err_after_sp_dens:.2f}")
# print(f"Within: {mean_within_sp_dens:.2f} +/- {std_err_within_sp_dens:.2f}")
# print(f"Between: {mean_between_sp_dens:.2f} +/- {std_err_between_sp_dens:.2f}")
#
# # Box plots for temp_dens
# plt.boxplot([cleaned_before_temp_dens, cleaned_after_temp_dens, cleaned_within_temp_dens, cleaned_between_temp_dens])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Temporal density differences among Ca2+ event groups')
# plt.ylabel('Temporal Density')
# plt.show()
#
# # Calculate and print mean +/- standard error for temp_dens
# mean_before_temp_dens, std_err_before_temp_dens = calculate_mean_std_err(cleaned_before_temp_dens)
# mean_after_temp_dens, std_err_after_temp_dens = calculate_mean_std_err(cleaned_after_temp_dens)
# mean_within_temp_dens, std_err_within_temp_dens = calculate_mean_std_err(cleaned_within_temp_dens)
# mean_between_temp_dens, std_err_between_temp_dens = calculate_mean_std_err(cleaned_between_temp_dens)
#
# print("Temporal Density:")
# print(f"Before: {mean_before_temp_dens:.2f} +/- {std_err_before_temp_dens:.2f}")
# print(f"After: {mean_after_temp_dens:.2f} +/- {std_err_after_temp_dens:.2f}")
# print(f"Within: {mean_within_temp_dens:.2f} +/- {std_err_within_temp_dens:.2f}")
# print(f"Between: {mean_between_temp_dens:.2f} +/- {std_err_between_temp_dens:.2f}")
#
# # Box plots for dur_10
# plt.boxplot([cleaned_before_dur_10, cleaned_after_dur_10, cleaned_within_dur_10, cleaned_between_dur_10])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Duration > 10 differences among Ca2+ event groups')
# plt.ylabel('Duration > 10')
# plt.show()
#
# # Calculate and print mean +/- standard error for dur_10
# mean_before_dur_10, std_err_before_dur_10 = calculate_mean_std_err(cleaned_before_dur_10)
# mean_after_dur_10, std_err_after_dur_10 = calculate_mean_std_err(cleaned_after_dur_10)
# mean_within_dur_10, std_err_within_dur_10 = calculate_mean_std_err(cleaned_within_dur_10)
# mean_between_dur_10, std_err_between_dur_10 = calculate_mean_std_err(cleaned_between_dur_10)
#
# print("Duration > 10:")
# print(f"Before: {mean_before_dur_10:.2f} +/- {std_err_before_dur_10:.2f}")
# print(f"After: {mean_after_dur_10:.2f} +/- {std_err_after_dur_10:.2f}")
# print(f"Within: {mean_within_dur_10:.2f} +/- {std_err_within_dur_10:.2f}")
# print(f"Between: {mean_between_dur_10:.2f} +/- {std_err_between_dur_10:.2f}")
# #
# plt.boxplot([cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Decay time differences among Ca2+ groups')
# plt.ylabel('Decay time (s)')
# plt.show()
# #
#
#
#
#
#
# #
# # # Calculate and print mean +/- standard error for rise time
# # mean_before_rise, std_err_before_rise = calculate_mean_std_err(cleaned_before_rise)
# # mean_after_rise, std_err_after_rise = calculate_mean_std_err(cleaned_after_rise)
# # mean_within_rise, std_err_within_rise = calculate_mean_std_err(cleaned_within_rise)
# # mean_between_rise, std_err_between_rise = calculate_mean_std_err(cleaned_between_rise)
# #
# # print("Rise time:")
# # print(f"Before: {mean_before_rise:.2f} +/- {std_err_before_rise:.2f}")
# # print(f"After: {mean_after_rise:.2f} +/- {std_err_after_rise:.2f}")
# # print(f"Within: {mean_within_rise:.2f} +/- {std_err_within_rise:.2f}")
# # print(f"Between: {mean_between_rise:.2f} +/- {std_err_between_rise:.2f}")
#
# Box plots for decay time
# plt.boxplot([cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay])
# plt.xticks([1, 2, 3, 4], ['Before', 'After', 'Within', 'Between'])
# plt.title('Decay time differences among Ca2+ groups')
# plt.ylabel('Decay time (s)')
# plt.show()
#






# ########################POST HOC ANALYSIS FOR SIGNIFICANT RESULTS#################################
import scikit_posthocs as sp
#
# data_propagation = [cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation]
# data_sp_dens = [cleaned_before_sp_dens, cleaned_after_sp_dens, cleaned_within_sp_dens, cleaned_between_sp_dens]
# data_temp_dens = [cleaned_before_temp_dens, cleaned_after_temp_dens, cleaned_within_temp_dens, cleaned_between_temp_dens]
# data_decay = [cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay]
#
# # Perform Dunn's test for the propagation data
# dunn_propagation = sp.posthoc_dunn(data_propagation)
#
# # Perform Dunn's test for the sp_dens data
# dunn_sp_dens = sp.posthoc_dunn(data_sp_dens)
#
# # Perform Dunn's test for the temp_dens data
# dunn_temp_dens = sp.posthoc_dunn(data_temp_dens)
#
# # Perform Dunn's test for the decay data
# dunn_decay = sp.posthoc_dunn(data_decay)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the propagation data
# significant_propagation = dunn_propagation < alpha
#
# # Identify significant pairwise comparisons for the sp_dens data
# significant_sp_dens = dunn_sp_dens < alpha
#
# # Identify significant pairwise comparisons for the temp_dens data
# significant_temp_dens = dunn_temp_dens < alpha
#
# # Identify significant pairwise comparisons for the decay data
# significant_decay = dunn_decay < alpha
#
# # Print the significant pairwise comparisons for the propagation data
# print("Significant pairwise comparisons for the propagation data:")
# print(significant_propagation)
#
# # Print the significant pairwise comparisons for the sp_dens data
# print("Significant pairwise comparisons for the sp_dens data:")
# print(significant_sp_dens)
#
# # Print the significant pairwise comparisons for the temp_dens data
# print("Significant pairwise comparisons for the temp_dens data:")
# print(significant_temp_dens)
#
# # Print the significant pairwise comparisons for the decay data
# print("Significant pairwise comparisons for the decay data:")
# print(significant_decay)
#
# # Print the Dunn's test results for the propagation data
# print("Dunn's test results for the propagation data:")
# print(dunn_propagation)
#
# # Print the Dunn's test results for the sp_dens data
# print("Dunn's test results for the sp_dens data:")
# print(dunn_sp_dens)
#
# # Print the Dunn's test results for the temp_dens data
# print("Dunn's test results for the temp_dens data:")
# print(dunn_temp_dens)
#
# # Print the Dunn's test results for the decay data
# print("Dunn's test results for the decay data:")
# print(dunn_decay)









#
# data_propagation = [cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation]
# data_sp_dens = [cleaned_before_sp_dens, cleaned_after_sp_dens, cleaned_within_sp_dens, cleaned_between_sp_dens]
# data_temp_dens = [cleaned_before_temp_dens, cleaned_after_temp_dens, cleaned_within_temp_dens, cleaned_between_temp_dens]
# data_decay = [cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay]
#
# # Perform Dunn's test for the propagation data
# dunn_propagation = sp.posthoc_dunn(data_propagation)
#
# # Perform Dunn's test for the sp_dens data
# dunn_sp_dens = sp.posthoc_dunn(data_sp_dens)
#
# # Perform Dunn's test for the temp_dens data
# dunn_temp_dens = sp.posthoc_dunn(data_temp_dens)
#
# # Perform Dunn's test for the decay data
# dunn_decay = sp.posthoc_dunn(data_decay)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the propagation data
# significant_propagation = dunn_propagation < alpha
#
# # Identify significant pairwise comparisons for the sp_dens data
# significant_sp_dens = dunn_sp_dens < alpha
#
# # Identify significant pairwise comparisons for the temp_dens data
# significant_temp_dens = dunn_temp_dens < alpha
#
# # Identify significant pairwise comparisons for the decay data
# significant_decay = dunn_decay < alpha
#
# # Print the significant pairwise comparisons for the propagation data
# print("Significant pairwise comparisons for the propagation data:")
# print(significant_propagation)
#
# # Print the significant pairwise comparisons for the sp_dens data
# print("Significant pairwise comparisons for the sp_dens data:")
# print(significant_sp_dens)
#
# # Print the significant pairwise comparisons for the temp_dens data
# print("Significant pairwise comparisons for the temp_dens data:")
# print(significant_temp_dens)
#
# # Print the significant pairwise comparisons for the decay data
# print("Significant pairwise comparisons for the decay data:")
# print(significant_decay)
#
# # Print the Dunn's test results for the propagation data
# print("Dunn's test results for the propagation data:")
# print(dunn_propagation)
#
# # Print the Dunn's test results for the sp_dens data
# print("Dunn's test results for the sp_dens data:")
# print(dunn_sp_dens)
#
# # Print the Dunn's test results for the temp_dens data
# print("Dunn's test results for the temp_dens data:")
# print(dunn_temp_dens)
#
# # Print the Dunn's test results for the decay data
# print("Dunn's test results for the decay data:")
# print(dunn_decay)

















######################## AREA BOXPLOTS WITH SIGNIFICANCE ########################

# import seaborn as sns
#
# # Combine the data into a single list
# data_areas = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
#
# # Perform Dunn's test for the areas data
# dunn_areas = sp.posthoc_dunn(data_areas)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the areas data
# significant_areas = dunn_areas < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_areas, showfliers=False)
#
# # Filter out the 'False' key from significant_areas
# significant_areas = significant_areas.loc[significant_areas.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start = max([max(area) for area in data_areas]) + 0.1
# y_shift = 3.5  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_areas.index):
#     for j, col in enumerate(significant_areas.columns):
#         if significant_areas.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start + (line_counter * y_shift)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_areas.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter += 1  # Increment the counter
#                     if line_counter >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels)), x_labels)
#
# # Set the y-axis label and plot title
# plt.ylabel('Area (μm**2)')
# plt.title('Area differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()
# #
#
# ####################### RISE BOXPLOTS WITH SIGNIFICANCE ########################
#
# import seaborn as sns
#
# # Combine the data into a single list
# data_rise = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
#
# # Perform Dunn's test for the rise data
# dunn_rise = sp.posthoc_dunn(data_rise)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the rise data
# significant_rise = dunn_rise < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_rise, showfliers=False)
#
# # Filter out the 'False' key from significant_rise
# significant_rise = significant_rise.loc[significant_rise.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start_rise = max([max(rise) for rise in data_rise]) + 0.1
# y_shift_rise = 2  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter_rise = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_rise.index):
#     for j, col in enumerate(significant_rise.columns):
#         if significant_rise.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start_rise + (line_counter_rise * y_shift_rise)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_rise.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter_rise += 1  # Increment the counter
#                     if line_counter_rise >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter_rise >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels)), x_labels)
#
# # Set the y-axis label and plot title
# plt.ylabel('Rise (μm)')
# plt.title('Rise differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()
#
# ########## FOR DURATION ######
#
#
# import seaborn as sns
#
# # Combine the data into a single list
# data_duration = [cleaned_before_dur, cleaned_after_dur, cleaned_within_dur, cleaned_between_dur]
#
# # Perform Dunn's test for the duration data
# dunn_duration = sp.posthoc_dunn(data_duration)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the duration data
# significant_duration = dunn_duration < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_duration, showfliers=False)
#
# # Filter out the 'False' key from significant_duration
# significant_duration = significant_duration.loc[significant_duration.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start_duration = max([max(duration) for duration in data_duration]) + 0.1
# y_shift_duration = 2  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter_duration = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_duration.index):
#     for j, col in enumerate(significant_duration.columns):
#         if significant_duration.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start_duration + (line_counter_duration * y_shift_duration)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_duration.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter_duration += 1  # Increment the counter
#                     if line_counter_duration >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter_duration >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels)), x_labels)
#
# # Set the y-axis label and plot title
# plt.ylabel('Duration (s)')
# plt.title('Duration differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()


############# FOR PROPAGATION #############

# import seaborn as sns
#
# # Combine the data into a single list
# data_propagation = [cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation]
#
# # Perform Dunn's test for the propagation data
# dunn_propagation = sp.posthoc_dunn(data_propagation)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the propagation data
# significant_propagation = dunn_propagation < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_propagation, showfliers=False)
#
# # Filter out the 'False' key from significant_propagation
# significant_propagation = significant_propagation.loc[significant_propagation.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start_propagation = max([max(propagation) for propagation in data_propagation]) + 0.1
# y_shift_propagation = 3.5  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter_propagation = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_propagation.index):
#     for j, col in enumerate(significant_propagation.columns):
#         if significant_propagation.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start_propagation + (line_counter_propagation * y_shift_propagation)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_propagation.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter_propagation += 1  # Increment the counter
#                     if line_counter_propagation >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter_propagation >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels_propagation = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels_propagation)), x_labels_propagation)
#
# # Set the y-axis label and plot title
# plt.ylabel('Propagation (μm/s)')
# plt.title('Propagation differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()



# import statsmodels.stats.multicomp as mc
#
# # Assuming you have your data stored in variables before_dur_flat, after_dur_flat, within_dur_flat, and between_dur_flat
#
# # Combine the data into a single array
# data_rise = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
# data_areas = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
#
# # Generate group labels based on the data
# labels_rise = ['before'] * len(cleaned_before_rise) + ['after'] * len(cleaned_after_rise) + ['within'] * len(cleaned_within_rise) + ['between'] * len(cleaned_between_rise)
# labels_areas = ['before'] * len(cleaned_before_areas) + ['after'] * len(cleaned_after_areas) + ['within'] * len(cleaned_within_areas) + ['between'] * len(cleaned_between_areas)
#
# # Create a MultiComparison object
# multi_comp_rise = mc.MultiComparison(np.concatenate(data_rise), labels_rise)
# multi_comp_areas = mc.MultiComparison(np.concatenate(data_areas), labels_areas)
#
# # Perform the post hoc analysis (Tukey's HSD test)
# result_rise = multi_comp_rise.tukeyhsd()
# result_areas = multi_comp_areas.tukeyhsd()
#
# # Print the results
# print(result)
#

#
# import seaborn as sns
#
# def add_significance_lines(significant, dunn, data):
#     y_start = max([max(d) for d in data]) + 0.1
#     y_shift = 3.5
#
#     line_counter = 0
#     for i, row in enumerate(significant.index):
#         for j, col in enumerate(significant.columns):
#             if significant.loc[row, col]:
#                 try:
#                     x1 = i
#                     x2 = j
#                     y = y_start + (line_counter * y_shift)
#                     if x1 < x2:
#                         plt.plot([x1, x2], [y, y], color='black', lw=1)
#                         p = dunn.loc[row, col]
#                         if p < 0.001:
#                             sig_symbol = '***'
#                         elif p < 0.01:
#                             sig_symbol = '**'
#                         elif p < 0.05:
#                             sig_symbol = '*'
#                         else:
#                             sig_symbol = ''
#                         plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                         line_counter += 1
#                         if line_counter >= 5:
#                             break
#                 except ValueError:
#                     pass
#         if line_counter >= 5:
#             break
#
# # Define the data for each variable
# data_rise_time = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
# data_area = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
# data_propagation = [cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation]
# data_sp_dens = [cleaned_before_sp_dens, cleaned_after_sp_dens, cleaned_within_sp_dens, cleaned_between_sp_dens]
# data_temp_dens = [cleaned_before_temp_dens, cleaned_after_temp_dens, cleaned_within_temp_dens, cleaned_between_temp_dens]
# data_decay = [cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay]
#
# # Perform Dunn's test for the rise_time data
# dunn_rise_time = sp.posthoc_dunn(data_rise_time)
#
# # Perform Dunn's test for the area data
# dunn_area = sp.posthoc_dunn(data_area)
#
# # Perform Dunn's test for the propagation data
# dunn_propagation = sp.posthoc_dunn(data_propagation)
#
# # Perform Dunn's test for the sp_dens data
# dunn_sp_dens = sp.posthoc_dunn(data_sp_dens)
#
# # Perform Dunn's test for the temp_dens data
# dunn_temp_dens = sp.posthoc_dunn(data_temp_dens)
#
# # Perform Dunn's test for the decay data
# dunn_decay = sp.posthoc_dunn(data_decay)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the rise_time data
# significant_rise_time = dunn_rise_time < alpha
#
# # Identify significant pairwise comparisons for the area data
# significant_area = dunn_area < alpha
#
# # Identify significant pairwise comparisons for the propagation data
# significant_propagation = dunn_propagation < alpha
#
# # Identify significant pairwise comparisons for the sp_dens data
# significant_sp_dens = dunn_sp_dens < alpha
#
# # Identify significant pairwise comparisons for the temp_dens data
# significant_temp_dens = dunn_temp_dens < alpha
#
# # Identify significant pairwise comparisons for the decay data
# significant_decay = dunn_decay < alpha
#
# # Plot boxplot without individual data points for rise_time
# sns.boxplot(data=data_rise_time, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Rise Time')
# plt.title('Rise Time differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_rise_time, dunn_rise_time, data_rise_time)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for area
# sns.boxplot(data=data_area, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Area')
# plt.title('Area differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_area, dunn_area, data_area)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for propagation
# sns.boxplot(data=data_propagation, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Propagation')
# plt.title('Propagation differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_propagation, dunn_propagation, data_propagation)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for sp_dens
# sns.boxplot(data=data_sp_dens, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Sp Dens')
# plt.title('Sp Dens differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_sp_dens, dunn_sp_dens, data_sp_dens)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for temp_dens
# sns.boxplot(data=data_temp_dens, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Temp Dens')
# plt.title('Temp Dens differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_temp_dens, dunn_temp_dens, data_temp_dens)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for decay
# sns.boxplot(data=data_decay, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Decay')
# plt.title('Decay differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_decay, dunn_decay, data_decay)
#
# # Show the plot
# plt.show()
#
#
# from scipy.stats import f_oneway
#
# # Define the data
# data = []
#
# # Check and add non-empty arrays to the data list
# if len(cleaned_before_rise) > 0:
#     data.append(cleaned_before_rise)
#     print("Number of calcium events before Sharp wave ripples:", len(cleaned_before_rise))
# if len(cleaned_within_rise) > 0:
#     data.append(cleaned_within_rise)
#     print("Number of calcium events within Sharp wave ripples:", len(cleaned_within_rise))
# if len(cleaned_after_rise) > 0:
#     data.append(cleaned_after_rise)
#     print("Number of calcium events after Sharp wave ripples:", len(cleaned_after_rise))
#
# # Perform one-way ANOVA if there is valid data
# if len(data) > 1:
#     f_statistic, p_value = f_oneway(*data)
#
#     # Set the significance level (alpha)
#     alpha = 0.05
#
#     # Compare the p-value with the significance level
#     if p_value < alpha:
#         print("The number of calcium events before Sharp wave ripples is significantly different.")
#     else:
#         print("The number of calcium events before Sharp wave ripples is not significantly different.")
# else:
#     print("Insufficient data to perform ANOVA.")



############################CHECKING THE PROPORTION OF OUTLIERS###################################

#
# m_outliers_removed_after_dur = len(after_dur_flat) - len(cleaned_after_dur)
# proportion_outliers_removed_after_dur = num_outliers_removed_after_dur / len(after_dur_flat)
# print(f"Proportion of outliers removed for after_dur_flat: {proportion_outliers_removed_after_dur:.2%}")
#
# # within_dur_flat
# num_outliers_removed_within_dur = len(within_dur_flat) - len(cleaned_within_dur)
# proportion_outliers_removed_within_dur = num_outliers_removed_within_dur / len(within_dur_flat)
# print(f"Proportion of outliers removed for within_dur_flat: {proportion_outliers_removed_within_dur:.2%}")
#
# # between_dur_flat
# num_outliers_removed_between_dur = len(between_dur_flat) - len(cleaned_between_dur)
# proportion_outliers_removed_between_dur = num_outliers_removed_between_dur / len(between_dur_flat)
# print(f"Proportion of outliers removed for between_dur_flat: {proportion_outliers_removed_between_dur:.2%}")
#
# # after_areas_flat
# num_outliers_removed_after_areas = len(after_areas_flat) - len(cleaned_after_areas)
# proportion_outliers_removed_after_areas = num_outliers_removed_after_areas / len(after_areas_flat)
# print(f"Proportion of outliers removed for after_areas_flat: {proportion_outliers_removed_after_areas:.2%}")
#
# # within_areas_flat
# num_outliers_removed_within_areas = len(within_areas_flat) - len(cleaned_within_areas)
# proportion_outliers_removed_within_areas = num_outliers_removed_within_areas / len(within_areas_flat)
# print(f"Proportion of outliers removed for within_areas_flat: {proportion_outliers_removed_within_areas:.2%}")
#
# # between_areas_flat
# num_outliers_removed_between_areas = len(between_areas_flat) - len(cleaned_between_areas)
# proportion_outliers_removed_between_areas = num_outliers_removed_between_areas / len(between_areas_flat)
# print(f"Proportion of outliers removed for between_areas_flat: {proportion_outliers_removed_between_areas:.2%}")
#
# # after_rise_flat
# num_outliers_removed_after_rise = len(after_rise_flat) - len(cleaned_after_rise)
# proportion_outliers_removed_after_rise = num_outliers_removed_after_rise / len(after_rise_flat)
# print(f"Proportion of outliers removed for after_rise_flat: {proportion_outliers_removed_after_rise:.2%}")
#
# # within_rise_flat
# num_outliers_removed_within_rise = len(within_rise_flat) - len(cleaned_within_rise)
# proportion_outliers_removed_within_rise = num_outliers_removed_within_rise / len(within_rise_flat)
# print(f"Proportion of outliers removed for within_rise_flat: {proportion_outliers_removed_within_rise:.2%}")
#
# # between_rise_flat
# num_outliers_removed_between_rise = len(between_rise_flat) - len(cleaned_between_rise)
# proportion_outliers_removed_between_rise = num_outliers_removed_between_rise / len(between_rise_flat)
# print(f"Proportion of outliers removed for between_rise_flat: {proportion_outliers_removed_between_rise:.2%}")
# after_dur_flat








######################## AREA BOXPLOTS WITH SIGNIFICANCE ########################

# import seaborn as sns
#
# # Combine the data into a single list
# data_areas = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
#
# # Perform Dunn's test for the areas data
# dunn_areas = sp.posthoc_dunn(data_areas)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the areas data
# significant_areas = dunn_areas < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_areas, showfliers=False)
#
# # Filter out the 'False' key from significant_areas
# significant_areas = significant_areas.loc[significant_areas.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start = max([max(area) for area in data_areas]) + 0.1
# y_shift = 3.5  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_areas.index):
#     for j, col in enumerate(significant_areas.columns):
#         if significant_areas.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start + (line_counter * y_shift)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_areas.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter += 1  # Increment the counter
#                     if line_counter >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels)), x_labels)
#
# # Set the y-axis label and plot title
# plt.ylabel('Area (μm**2)')
# plt.title('Area differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()
#
#
# ####################### RISE BOXPLOTS WITH SIGNIFICANCE ########################
#
# import seaborn as sns
#
# # Combine the data into a single list
# data_rise = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
#
# # Perform Dunn's test for the rise data
# dunn_rise = sp.posthoc_dunn(data_rise)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the rise data
# significant_rise = dunn_rise < alpha
#
# # Plot boxplot without individual data points
# sns.boxplot(data=data_rise, showfliers=False)
#
# # Filter out the 'False' key from significant_rise
# significant_rise = significant_rise.loc[significant_rise.index != False]
#
# # Calculate the y-coordinate for the lines and asterisks
# y_start_rise = max([max(rise) for rise in data_rise]) + 0.1
# y_shift_rise = 2  # Increase this value to create a larger vertical gap between lines
#
# # Add horizontal lines and asterisks for each significant group
# line_counter_rise = 0  # Counter variable to track the number of lines plotted
# for i, row in enumerate(significant_rise.index):
#     for j, col in enumerate(significant_rise.columns):
#         if significant_rise.loc[row, col]:
#             try:
#                 # Get the x-coordinates for the bars
#                 x1 = i
#                 x2 = j
#                 # Calculate the y-coordinate for the line and text
#                 y = y_start_rise + (line_counter_rise * y_shift_rise)
#                 # Plot the horizontal line
#                 if x1 < x2:  # Add condition to plot line only once between groups
#                     plt.plot([x1, x2], [y, y], color='black', lw=1)
#                     # Add the significance annotation
#                     p = dunn_rise.loc[row, col]
#                     if p < 0.001:
#                         sig_symbol = '***'
#                     elif p < 0.01:
#                         sig_symbol = '**'
#                     elif p < 0.05:
#                         sig_symbol = '*'
#                     else:
#                         sig_symbol = ''
#                     plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                     line_counter_rise += 1  # Increment the counter
#                     if line_counter_rise >= 3:  # Break out of the loop after plotting three lines
#                         break
#             except ValueError:
#                 pass
#     if line_counter_rise >= 3:  # Break out of the loop after plotting three lines
#         break
#
# # Set the x-axis labels
# x_labels = ['Before', 'After', 'Within', 'Between']
# plt.xticks(range(len(x_labels)), x_labels)
#
# # Set the y-axis label and plot title
# plt.ylabel('Rise (μm)')
# plt.title('Rise differences among Ca2+ event groups')
#
# # Show the plot
# plt.show()





# import statsmodels.stats.multicomp as mc
#
# # Assuming you have your data stored in variables before_dur_flat, after_dur_flat, within_dur_flat, and between_dur_flat
#
# # Combine the data into a single array
# data_rise = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
# data_areas = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
#
# # Generate group labels based on the data
# labels_rise = ['before'] * len(cleaned_before_rise) + ['after'] * len(cleaned_after_rise) + ['within'] * len(cleaned_within_rise) + ['between'] * len(cleaned_between_rise)
# labels_areas = ['before'] * len(cleaned_before_areas) + ['after'] * len(cleaned_after_areas) + ['within'] * len(cleaned_within_areas) + ['between'] * len(cleaned_between_areas)
#
# # Create a MultiComparison object
# multi_comp_rise = mc.MultiComparison(np.concatenate(data_rise), labels_rise)
# multi_comp_areas = mc.MultiComparison(np.concatenate(data_areas), labels_areas)
#
# # Perform the post hoc analysis (Tukey's HSD test)
# result_rise = multi_comp_rise.tukeyhsd()
# result_areas = multi_comp_areas.tukeyhsd()
#
# # Print the results
# print(result)
#

#
# import seaborn as sns
#
# def add_significance_lines(significant, dunn, data):
#     y_start = max([max(d) for d in data]) + 0.1
#     y_shift = 3.5
#
#     line_counter = 0
#     for i, row in enumerate(significant.index):
#         for j, col in enumerate(significant.columns):
#             if significant.loc[row, col]:
#                 try:
#                     x1 = i
#                     x2 = j
#                     y = y_start + (line_counter * y_shift)
#                     if x1 < x2:
#                         plt.plot([x1, x2], [y, y], color='black', lw=1)
#                         p = dunn.loc[row, col]
#                         if p < 0.001:
#                             sig_symbol = '***'
#                         elif p < 0.01:
#                             sig_symbol = '**'
#                         elif p < 0.05:
#                             sig_symbol = '*'
#                         else:
#                             sig_symbol = ''
#                         plt.text((x1 + x2) * 0.5, y, sig_symbol, ha='center', va='bottom')
#                         line_counter += 1
#                         if line_counter >= 5:
#                             break
#                 except ValueError:
#                     pass
#         if line_counter >= 5:
#             break
#
# # Define the data for each variable
# data_rise_time = [cleaned_before_rise, cleaned_after_rise, cleaned_within_rise, cleaned_between_rise]
# data_area = [cleaned_before_areas, cleaned_after_areas, cleaned_within_areas, cleaned_between_areas]
# data_propagation = [cleaned_before_propagation, cleaned_after_propagation, cleaned_within_propagation, cleaned_between_propagation]
# data_sp_dens = [cleaned_before_sp_dens, cleaned_after_sp_dens, cleaned_within_sp_dens, cleaned_between_sp_dens]
# data_temp_dens = [cleaned_before_temp_dens, cleaned_after_temp_dens, cleaned_within_temp_dens, cleaned_between_temp_dens]
# data_decay = [cleaned_before_decay, cleaned_after_decay, cleaned_within_decay, cleaned_between_decay]
#
# # Perform Dunn's test for the rise_time data
# dunn_rise_time = sp.posthoc_dunn(data_rise_time)
#
# # Perform Dunn's test for the area data
# dunn_area = sp.posthoc_dunn(data_area)
#
# # Perform Dunn's test for the propagation data
# dunn_propagation = sp.posthoc_dunn(data_propagation)
#
# # Perform Dunn's test for the sp_dens data
# dunn_sp_dens = sp.posthoc_dunn(data_sp_dens)
#
# # Perform Dunn's test for the temp_dens data
# dunn_temp_dens = sp.posthoc_dunn(data_temp_dens)
#
# # Perform Dunn's test for the decay data
# dunn_decay = sp.posthoc_dunn(data_decay)
#
# # Set the significance level (alpha)
# alpha = 0.05
#
# # Identify significant pairwise comparisons for the rise_time data
# significant_rise_time = dunn_rise_time < alpha
#
# # Identify significant pairwise comparisons for the area data
# significant_area = dunn_area < alpha
#
# # Identify significant pairwise comparisons for the propagation data
# significant_propagation = dunn_propagation < alpha
#
# # Identify significant pairwise comparisons for the sp_dens data
# significant_sp_dens = dunn_sp_dens < alpha
#
# # Identify significant pairwise comparisons for the temp_dens data
# significant_temp_dens = dunn_temp_dens < alpha
#
# # Identify significant pairwise comparisons for the decay data
# significant_decay = dunn_decay < alpha
#
# # Plot boxplot without individual data points for rise_time
# sns.boxplot(data=data_rise_time, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Rise Time')
# plt.title('Rise Time differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_rise_time, dunn_rise_time, data_rise_time)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for area
# sns.boxplot(data=data_area, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Area')
# plt.title('Area differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_area, dunn_area, data_area)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for propagation
# sns.boxplot(data=data_propagation, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Propagation')
# plt.title('Propagation differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_propagation, dunn_propagation, data_propagation)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for sp_dens
# sns.boxplot(data=data_sp_dens, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Sp Dens')
# plt.title('Sp Dens differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_sp_dens, dunn_sp_dens, data_sp_dens)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for temp_dens
# sns.boxplot(data=data_temp_dens, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Temp Dens')
# plt.title('Temp Dens differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_temp_dens, dunn_temp_dens, data_temp_dens)
#
# # Show the plot
# plt.show()
#
# # Plot boxplot without individual data points for decay
# sns.boxplot(data=data_decay, showfliers=False)
# plt.xticks(range(4), ['Before', 'After', 'Within', 'Between'])
# plt.ylabel('Decay')
# plt.title('Decay differences among Ca2+ event groups')
#
# # Add horizontal lines and asterisks for significant comparisons
# add_significance_lines(significant_decay, dunn_decay, data_decay)
#
# # Show the plot
# plt.show()
#
#
# from scipy.stats import f_oneway
#
# # Define the data
# data = []
#
# # Check and add non-empty arrays to the data list
# if len(cleaned_before_rise) > 0:
#     data.append(cleaned_before_rise)
#     print("Number of calcium events before Sharp wave ripples:", len(cleaned_before_rise))
# if len(cleaned_within_rise) > 0:
#     data.append(cleaned_within_rise)
#     print("Number of calcium events within Sharp wave ripples:", len(cleaned_within_rise))
# if len(cleaned_after_rise) > 0:
#     data.append(cleaned_after_rise)
#     print("Number of calcium events after Sharp wave ripples:", len(cleaned_after_rise))
#
# # Perform one-way ANOVA if there is valid data
# if len(data) > 1:
#     f_statistic, p_value = f_oneway(*data)
#
#     # Set the significance level (alpha)
#     alpha = 0.05
#
#     # Compare the p-value with the significance level
#     if p_value < alpha:
#         print("The number of calcium events before Sharp wave ripples is significantly different.")
#     else:
#         print("The number of calcium events before Sharp wave ripples is not significantly different.")
# else:
#     print("Insufficient data to perform ANOVA.")



############################CHECKING THE PROPORTION OF OUTLIERS###################################

#
# m_outliers_removed_after_dur = len(after_dur_flat) - len(cleaned_after_dur)
# proportion_outliers_removed_after_dur = num_outliers_removed_after_dur / len(after_dur_flat)
# print(f"Proportion of outliers removed for after_dur_flat: {proportion_outliers_removed_after_dur:.2%}")
#
# # within_dur_flat
# num_outliers_removed_within_dur = len(within_dur_flat) - len(cleaned_within_dur)
# proportion_outliers_removed_within_dur = num_outliers_removed_within_dur / len(within_dur_flat)
# print(f"Proportion of outliers removed for within_dur_flat: {proportion_outliers_removed_within_dur:.2%}")
#
# # between_dur_flat
# num_outliers_removed_between_dur = len(between_dur_flat) - len(cleaned_between_dur)
# proportion_outliers_removed_between_dur = num_outliers_removed_between_dur / len(between_dur_flat)
# print(f"Proportion of outliers removed for between_dur_flat: {proportion_outliers_removed_between_dur:.2%}")
#
# # after_areas_flat
# num_outliers_removed_after_areas = len(after_areas_flat) - len(cleaned_after_areas)
# proportion_outliers_removed_after_areas = num_outliers_removed_after_areas / len(after_areas_flat)
# print(f"Proportion of outliers removed for after_areas_flat: {proportion_outliers_removed_after_areas:.2%}")
#
# # within_areas_flat
# num_outliers_removed_within_areas = len(within_areas_flat) - len(cleaned_within_areas)
# proportion_outliers_removed_within_areas = num_outliers_removed_within_areas / len(within_areas_flat)
# print(f"Proportion of outliers removed for within_areas_flat: {proportion_outliers_removed_within_areas:.2%}")
#
# # between_areas_flat
# num_outliers_removed_between_areas = len(between_areas_flat) - len(cleaned_between_areas)
# proportion_outliers_removed_between_areas = num_outliers_removed_between_areas / len(between_areas_flat)
# print(f"Proportion of outliers removed for between_areas_flat: {proportion_outliers_removed_between_areas:.2%}")
#
# # after_rise_flat
# num_outliers_removed_after_rise = len(after_rise_flat) - len(cleaned_after_rise)
# proportion_outliers_removed_after_rise = num_outliers_removed_after_rise / len(after_rise_flat)
# print(f"Proportion of outliers removed for after_rise_flat: {proportion_outliers_removed_after_rise:.2%}")
#
# # within_rise_flat
# num_outliers_removed_within_rise = len(within_rise_flat) - len(cleaned_within_rise)
# proportion_outliers_removed_within_rise = num_outliers_removed_within_rise / len(within_rise_flat)
# print(f"Proportion of outliers removed for within_rise_flat: {proportion_outliers_removed_within_rise:.2%}")
#
# # between_rise_flat
# num_outliers_removed_between_rise = len(between_rise_flat) - len(cleaned_between_rise)
# proportion_outliers_removed_between_rise = num_outliers_removed_between_rise / len(between_rise_flat)
# print(f"Proportion of outliers removed for between_rise_flat: {proportion_outliers_removed_between_rise:.2%}")
# after_dur_flat



