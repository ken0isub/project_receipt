import os
import cv2
import numpy as np
import pickle
import pandas as pd
from data_prep import img_prep


def validate_ml(model_path):
    classes = []
    with open('./models/stores_list.txt', 'r') as f:
        for c in f:
            classes.append(c.rstrip())

    res_summary = []
    for model in os.listdir(model_path):
        with open(model_path + '/' + model, mode='rb') as fp:
            clf = pickle.load(fp)

        val_data_path = './validation'
        files = os.listdir(val_data_path)
        if 'desktop.ini' in files:
            files.remove('desktop.ini')
        else:
            pass

        correct = 0
        res = []

        for i, file in enumerate(files):
            val_img = cv2.imread('./validation/' + file)
            val_img = img_prep(val_img, gray_scale=True)
            X_sample = np.array(val_img)
            X_sample = X_sample.flatten()

            answer = clf.predict(X_sample.reshape(1, -1))[0]

            if classes[answer] == file[:-7]:
                correct += 1
            else:
                pass

            res.append([str(model), file[:-7], classes[answer]])
            pd.DataFrame(res).to_csv('models/summary/' + str(model[:-7]) + '_res.csv')

        res_summary.append([str(model), round(correct/len(files), 3)])
    return res_summary
