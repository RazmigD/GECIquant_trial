import pandas as pd
import numpy as np
import os
import pickle

data_folder = r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/data'
subfolders = [f.name for f in os.scandir(data_folder) if f.is_dir()]

# Split subfolder names into parts, sort based on numbers, and join back together
subfolders = sorted(subfolders, key=lambda x: [int(part) if part.isdigit() else part for part in x.split('_')])



df_summary = pd.DataFrame(index=range(len(subfolders)), columns=['subfolder_name'])
df_summary['subfolder_name'] = subfolders
df_summary.to_csv('df_summary.csv', index=False)

# # GENERATING EMPTY FOLDERS TO GET CA EVENT INDEXES (4 groups) FOR CORRELATION
# before_indexes = [[] for _ in range(len(df_summary))]
# within_indexes = [[] for _ in range(len(df_summary))]
# after_indexes = [[] for _ in range(len(df_summary))]
# between_indexes = [[] for _ in range(len(df_summary))]
# #
# #
# with open('before_indexes.pkl', 'wb') as f:
#     pickle.dump(before_indexes, f)
#
# with open('within_indexes.pkl', 'wb') as f:
#     pickle.dump(within_indexes, f)
#
# with open('after_indexes.pkl', 'wb') as f:
#     pickle.dump(after_indexes, f)
#
# with open('between_indexes.pkl', 'wb') as f:
#     pickle.dump(between_indexes, f)
#
# print(df_summary)
# print('before indexes:', before_indexes)
# print('within indexes:', within_indexes)
# print('after indexes:', after_indexes)
# print('between indexes:', between_indexes)



# #
# ######################################

before_areas = [[] for _ in range(len(subfolders))]
after_areas = [[] for _ in range(len(subfolders))]
within_areas = [[] for _ in range(len(subfolders))]
between_areas = [[] for _ in range(len(subfolders))]

with open('before_areas.pkl', 'wb') as f:
    pickle.dump(before_areas, f)

with open('within_areas.pkl', 'wb') as f:
    pickle.dump(within_areas, f)

with open('after_areas.pkl', 'wb') as f:
    pickle.dump(after_areas, f)

with open('between_areas.pkl', 'wb') as f:
    pickle.dump(between_areas, f)

#################################

before_dur = [[] for _ in range(len(subfolders))]
after_dur = [[] for _ in range(len(subfolders))]
within_dur = [[] for _ in range(len(subfolders))]
between_dur = [[] for _ in range(len(subfolders))]

with open('before_dur.pkl', 'wb') as f:
    pickle.dump(before_dur, f)

with open('within_dur.pkl', 'wb') as f:
    pickle.dump(after_dur, f)

with open('after_dur.pkl', 'wb') as f:
    pickle.dump(within_dur, f)

with open('between_dur.pkl', 'wb') as f:
    pickle.dump(between_dur, f)

###################################

before_dff = [[] for _ in range(len(subfolders))]
after_dff = [[] for _ in range(len(subfolders))]
within_dff = [[] for _ in range(len(subfolders))]
between_dff = [[] for _ in range(len(subfolders))]

with open('before_dff.pkl', 'wb') as f:
    pickle.dump(before_dff, f)

with open('within_dff.pkl', 'wb') as f:
    pickle.dump(within_dff, f)

with open('after_dff.pkl', 'wb') as f:
    pickle.dump(after_dff, f)

with open('between_dff.pkl', 'wb') as f:
    pickle.dump(between_dff, f)

##############################

before_rise = [[] for _ in range(len(df_summary))]
within_rise = [[] for _ in range(len(df_summary))]
after_rise = [[] for _ in range(len(df_summary))]
between_rise = [[] for _ in range(len(df_summary))]


with open('before_rise.pkl', 'wb') as f:
    pickle.dump(before_rise, f)

with open('within_rise.pkl', 'wb') as f:
    pickle.dump(within_rise, f)

with open('after_rise.pkl', 'wb') as f:
    pickle.dump(after_rise, f)

with open('between_rise.pkl', 'wb') as f:
    pickle.dump(between_rise, f)

################################

before_decay = [[] for _ in range(len(df_summary))]
within_decay = [[] for _ in range(len(df_summary))]
after_decay = [[] for _ in range(len(df_summary))]
between_decay = [[] for _ in range(len(df_summary))]


with open('before_decay.pkl', 'wb') as f:
    pickle.dump(before_decay, f)

with open('within_decay.pkl', 'wb') as f:
    pickle.dump(within_decay, f)

with open('after_decay.pkl', 'wb') as f:
    pickle.dump(after_decay, f)

with open('between_decay.pkl', 'wb') as f:
    pickle.dump(between_decay, f)

########################################

before_propagation = [[] for _ in range(len(df_summary))]
within_propagation = [[] for _ in range(len(df_summary))]
after_propagation = [[] for _ in range(len(df_summary))]
between_propagation = [[] for _ in range(len(df_summary))]

with open('before_propagation.pkl', 'wb') as f:
    pickle.dump(before_propagation, f)

with open('within_propagation.pkl', 'wb') as f:
    pickle.dump(within_propagation, f)

with open('after_propagation.pkl', 'wb') as f:
    pickle.dump(after_propagation, f)

with open('between_propagation.pkl', 'wb') as f:
    pickle.dump(between_propagation, f)

########################################

before_sp_dens = [[] for _ in range(len(df_summary))]
after_sp_dens = [[] for _ in range(len(df_summary))]
within_sp_dens = [[] for _ in range(len(df_summary))]
between_sp_dens = [[] for _ in range(len(df_summary))]

with open('before_sp_dens.pkl', 'wb') as f:
    pickle.dump(before_sp_dens, f)

with open('after_sp_dens.pkl', 'wb') as f:
    pickle.dump(after_sp_dens, f)

with open('within_sp_dens.pkl', 'wb') as f:
    pickle.dump(within_sp_dens, f)

with open('between_sp_dens.pkl', 'wb') as f:
    pickle.dump(between_sp_dens, f)

######################################

before_temp_dens = [[] for _ in range(len(df_summary))]
after_temp_dens = [[] for _ in range(len(df_summary))]
within_temp_dens = [[] for _ in range(len(df_summary))]
between_temp_dens = [[] for _ in range(len(df_summary))]

with open('before_temp_dens.pkl', 'wb') as f:
    pickle.dump(before_temp_dens, f)

with open('after_temp_dens.pkl', 'wb') as f:
    pickle.dump(after_temp_dens, f)

with open('within_temp_dens.pkl', 'wb') as f:
    pickle.dump(within_temp_dens, f)

with open('between_temp_dens.pkl', 'wb') as f:
    pickle.dump(between_temp_dens, f)


#######################################

before_dur_10 = [[] for _ in range(len(df_summary))]
after_dur_10 = [[] for _ in range(len(df_summary))]
within_dur_10 = [[] for _ in range(len(df_summary))]
between_dur_10 = [[] for _ in range(len(df_summary))]

with open('before_dur_10.pkl', 'wb') as f:
    pickle.dump(before_dur_10, f)

with open('after_dur_10.pkl', 'wb') as f:
    pickle.dump(after_dur_10, f)

with open('within_dur_10.pkl', 'wb') as f:
    pickle.dump(within_dur_10, f)

with open('between_dur_10.pkl', 'wb') as f:
    pickle.dump(between_dur_10, f)











# SPW_frequencies = []
# ca_event_num = []
#
# with open('SPW_frequencies.pkl', 'wb') as f:
#     pickle.dump(SPW_frequencies, f)
#
# with open('ca_event_num.pkl', 'wb') as f:
#     pickle.dump(ca_event_num, f)

#-----------------------------
#

