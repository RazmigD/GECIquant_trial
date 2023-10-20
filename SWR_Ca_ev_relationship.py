import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
from scipy.stats import f_oneway

# df_summary = pd.read_csv('df_summary.csv')
#
# print(df_summary)
# import pickle
# with open('SPW_frequencies.pkl', 'rb') as f:
#     SPW_frequencies = pickle.load(f)
#
# with open('ca_event_num.pkl', 'rb') as f:
#     ca_event_num = pickle.load(f)
#
#
# calcium_event_num = SPW_frequencies
# swr_freq = ca_event_num
#
#
# # Perform linear regression
# regression_line = np.polyfit(calcium_event_num, swr_freq, 1)
# line_fn = np.poly1d(regression_line)
#
# # Scatter plot
# plt.scatter(calcium_event_num, swr_freq, color='blue', label='Data Points')
#
# # Regression line
# plt.plot(calcium_event_num, line_fn(calcium_event_num), color='red', linestyle='-', label='Regression Line')
#
# plt.xlabel('Calcium Event Number')
# plt.ylabel('SWR Frequency')
# plt.title('Relationship between Calcium Event Number and SWR Frequency')
#
# plt.legend()
# # Remove upper and right axes spines
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
#
# correlation_coeff, p_value = stats.pearsonr(swr_freq, calcium_event_num)
#
# plt.text(min(calcium_event_num), max(swr_freq), f"Correlation: {correlation_coeff:.2f}\n p-value: {p_value}",
#          fontsize=10, color='black', verticalalignment='top', horizontalalignment='left')
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust plot layout to make room for text
#
# plt.show()

################# Correlation with average parameter values ##############

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr, f_oneway

# Remove outliers from each sublist
def remove_outliers_iqr(sublist, k=1.5):
    if len(sublist) > 0:
        q1 = np.percentile(sublist, 25)
        q3 = np.percentile(sublist, 75)
        iqr_value = iqr(sublist)
        lower_bound = q1 - k * iqr_value
        upper_bound = q3 + k * iqr_value
        filtered_sublist = [value for value in sublist if lower_bound <= value <= upper_bound]
        return filtered_sublist
    else:
        return []

# Remove values above 50 from each sublist
def remove_values_above_threshold(sublist, threshold=50):
    filtered_sublist = [value for value in sublist if value <= threshold]
    return filtered_sublist

before_dur_no_outliers = [remove_outliers_iqr(sublist) for sublist in before_dur]
after_dur_no_outliers = [remove_outliers_iqr(sublist) for sublist in after_dur]
between_dur_no_outliers = [remove_outliers_iqr(sublist) for sublist in between_dur]
within_dur_no_outliers = [remove_outliers_iqr(sublist) for sublist in within_dur]

before_areas_no_outliers = [remove_values_above_threshold(sublist) for sublist in before_areas]
after_areas_no_outliers = [remove_values_above_threshold(sublist) for sublist in after_areas]
between_areas_no_outliers = [remove_values_above_threshold(sublist) for sublist in between_areas]
within_areas_no_outliers = [remove_values_above_threshold(sublist) for sublist in within_areas]

before_rise_no_outliers = [remove_outliers_iqr(sublist) for sublist in before_rise]
after_rise_no_outliers = [remove_outliers_iqr(sublist) for sublist in after_rise]
between_rise_no_outliers = [remove_outliers_iqr(sublist) for sublist in between_rise]
within_rise_no_outliers = [remove_outliers_iqr(sublist) for sublist in within_rise]

# Calculate means
before_dur_means = [np.mean(sublist) for sublist in before_dur_no_outliers if len(sublist) > 0]
after_dur_means = [np.mean(sublist) for sublist in after_dur_no_outliers if len(sublist) > 0]
between_dur_means = [np.mean(sublist) for sublist in between_dur_no_outliers if len(sublist) > 0]
within_dur_means = [np.mean(sublist) for sublist in within_dur_no_outliers if len(sublist) > 0]

before_areas_means = [np.mean(sublist) for sublist in before_areas_no_outliers if len(sublist) > 0]
after_areas_means = [np.mean(sublist) for sublist in after_areas_no_outliers if len(sublist) > 0]
between_areas_means = [np.mean(sublist) for sublist in between_areas_no_outliers if len(sublist) > 0]
within_areas_means = [np.mean(sublist) for sublist in within_areas_no_outliers if len(sublist) > 0]

before_rise_means = [np.mean(sublist) for sublist in before_rise_no_outliers if len(sublist) > 0]
after_rise_means = [np.mean(sublist) for sublist in after_rise_no_outliers if len(sublist) > 0]
between_rise_means = [np.mean(sublist) for sublist in between_rise_no_outliers if len(sublist) > 0]
within_rise_means = [np.mean(sublist) for sublist in within_rise_no_outliers if len(sublist) > 0]

# Combine mean values into a single list
all_dur_means = [before_dur_means, within_dur_means, after_dur_means, between_dur_means]
all_areas_means = [before_areas_means, within_areas_means, after_areas_means, between_areas_means]
all_rise_means = [before_rise_means, within_rise_means, after_rise_means, between_rise_means]


##########################

# Create x-axis positions
x_pos = np.arange(len(all_dur_means))

# Create histogram for duration
plt.figure(figsize=(10, 6))
for i, means in enumerate(all_dur_means):
    plt.bar(x_pos[i], np.mean(means), yerr=np.std(means), align='center', alpha=0.5, capsize=10)  # Plot mean with error bars
plt.xticks(x_pos, ['Before', 'Within', 'After', 'Between'])
plt.xlabel('Groups')
plt.ylabel('Duration')
plt.title('Histogram of Duration')

# Perform ANOVA analysis for duration means
f_statistic_dur, p_value_dur = f_oneway(*all_dur_means)
print("Duration - F-Statistic:", f_statistic_dur)
print("Duration - P-Value:", p_value_dur)

plt.show()

########################

# Create x-axis positions
x_pos = np.arange(len(all_areas_means))

# Create histogram for areas
plt.figure(figsize=(10, 6))
for i, means in enumerate(all_areas_means):
    plt.bar(x_pos[i], np.mean(means), yerr=np.std(means), align='center', alpha=0.5, capsize=10)  # Plot mean with error bars
plt.xticks(x_pos, ['Before', 'Within', 'After', 'Between'])
plt.xlabel('Groups')
plt.ylabel('Area')
plt.title('Histogram of Area')

# Perform one-way ANOVA analysis on all_areas_means
f_statistic_areas, p_value_areas = f_oneway(*all_areas_means)
print("Area - F-Statistic:", f_statistic_areas)
print("Area - P-Value:", p_value_areas)

plt.show()

########################

# Create x-axis positions
x_pos = np.arange(len(all_rise_means))

# Create histogram for rise
plt.figure(figsize=(10, 6))
for i, means in enumerate(all_rise_means):
    plt.bar(x_pos[i], np.mean(means), yerr=np.std(means), align='center', alpha=0.5, capsize=10)  # Plot mean with error bars
plt.xticks(x_pos, ['Before', 'Within', 'After', 'Between'])
plt.xlabel('Groups')
plt.ylabel('Rise')
plt.title('Histogram of Rise')

# Perform one-way ANOVA analysis on all_rise_means
f_statistic_rise, p_value_rise = f_oneway(*all_rise_means)
print("Rise - F-Statistic:", f_statistic_rise)
print("Rise - P-Value:", p_value_rise)

plt.show()





