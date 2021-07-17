import os
import cv2
import numpy as np
from data_prep import img_prep
import pandas as pd

scan_data_path = 'training'

files = list(os.listdir(scan_data_path))
if 'desktop.ini' in files:
    files.remove('desktop.ini')
else:
    pass

template_data = {}
for file in files:
    img = cv2.imread(scan_data_path + '/' + file)
    template_data[file[:-7]] = img_prep(img, gray_scale=True)


validation_data_path = './validation'
val_files = list(os.listdir(validation_data_path))
if 'desktop.ini' in val_files:
    val_files.remove('desktop.ini')
else:
    pass

res_summary = []
for val_file in val_files:
    img = cv2.imread(validation_data_path + '/' + val_file)

    identities = ['dummy', 0]
    for store in template_data.keys():
        diff = np.count_nonzero(template_data[store] - img_prep(img, gray_scale=True)) / template_data[store].size
        identity = 1 - diff
        if identity > identities[1]:
            identities = [store, identity]
    if val_file[:-7] == identities[0]:
        identities.append('True')
    else:
        identities.append('False')
    res_summary.append(identities)


res_details = []
for val_file in val_files:
    img = cv2.imread(validation_data_path + '/' + val_file)
    diff = np.count_nonzero(template_data['seven'] - img_prep(img, gray_scale=True)) / template_data['seven'].size
    identity = 1 - diff
    res_details.append([val_file[:-7], identity])


print(res_summary)
df = pd.DataFrame(res_summary)
print(df)
df.to_csv('./models/summary/identity.csv')


print(res_details)
df_detail = pd.DataFrame(res_details)
print(df_detail)
df_detail.to_csv('./mdoels/summary/identity_details.csv')
