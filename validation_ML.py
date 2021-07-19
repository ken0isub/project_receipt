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
    ml_scores = []
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
        ml_scores.append(round(correct/len(files), 3))

    with open('./models/ml_scores.txt', 'w') as f:
        for c in ml_scores:
            print(c, file=f)

    return res_summary


def model_prediction(model_path, file_path):
    with open(model_path, mode ='rb') as fp:
        clf = pickle.load(fp)
    img = cv2.imread(file_path)
    img = img_prep(img, gray_scale=True)
    X_sample = np.array(img)
    X_sample = X_sample.flatten()
    return clf.predict(X_sample.reshape(1, -1))[0]


def ml_combined(file_path, model_path):
    classes = []
    with open('./models/stores_list.txt', 'r') as f:
        for c in f:
            classes.append(c.rstrip())

    ml_scores = []
    with open('models/ml_scores.txt', 'r') as f:
        for c in f:
            ml_scores.append(c.rstrip())

    models_list = os.listdir(model_path)

    prediction_1 = model_prediction(model_path + '/' + models_list[0], file_path)
    prediction_2 = model_prediction(model_path + '/' + models_list[1], file_path)
    prediction_3 = model_prediction(model_path + '/' + models_list[2], file_path)
    prediction_4 = model_prediction(model_path + '/' + models_list[3], file_path)
    prediction_5 = model_prediction(model_path + '/' + models_list[4], file_path)

    n_1 = np.zeros(len(classes))
    n_1[prediction_1] = ml_scores[0]

    n_2 = np.zeros(len(classes))
    n_2[prediction_2] = ml_scores[1]

    n_3 = np.zeros(len(classes))
    n_3[prediction_3] = ml_scores[2]

    n_4 = np.zeros(len(classes))
    n_4[prediction_4] = ml_scores[3]

    n_5 = np.zeros(len(classes))
    n_5[prediction_5] = ml_scores[4]

    pred_combined = n_1 + n_2 + n_3 + n_4 + n_5
    return classes[np.argmax(pred_combined)]


def validate_combined_ml(model_path):
    res_summary = []
    correct = 0
    res = []

    val_data_path = './validation'
    files = os.listdir(val_data_path)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    else:
        pass

    for file in files:
        answer = ml_combined(val_data_path + '/' + file, model_path)
        if answer == file[:-7]:
            correct += 1
        else:
            pass
        res.append(['combined_ml', file[:-7], answer])

    pd.DataFrame(res).to_csv('models/summary/combined_ml_res.csv')
    res_summary.append(['combined_ml', round(correct/len(files), 3)])
    return res_summary
