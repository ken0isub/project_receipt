import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_prep import img_prep
import pandas as pd


def validate_cnn(model, validation_dir):
    model = load_model('models/CNN/' + model + '.h5')

    classes = []
    with open('./models/stores_list.txt', 'r') as f:
        for c in f:
            classes.append(c.rstrip())

    files = os.listdir(validation_dir)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    else:
        pass

    correct = 0
    res = []

    for i, file in enumerate(files):
        val_img = cv2.imread(validation_dir + '/' + file)
        val_img = img_prep(val_img, gray_scale=False)
        X_sample = np.array(val_img)
        X_sample = X_sample.reshape(-1, 148, 298, 3)

        # print(str(model.predict(X_sample)))
        answer = np.argmax(model.predict(X_sample), axis=1)
        confidence = np.max(model.predict(X_sample), axis=1)
        print(f'[{i}] prediction: \"{classes[answer[0]]}\" with the confidence of {confidence}')

        if classes[answer[0]] == file[:-7]:
            print('correct!')
            correct += 1
        else:
            print(f'wrong, it\'s {file[:-7]}')

        res.append(['CNN', file[:-7], classes[answer[0]]])

        print('=' *60)

    print(f'accuracy: {correct/len(files)}')
    pd.DataFrame(res).to_csv('./models/summary/CNN_res.csv')