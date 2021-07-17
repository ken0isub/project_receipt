import cv2
import os
import matplotlib.pyplot as plt
from data_prep import img_prep
import pandas as pd

def similarity_search(target_img_dir, target_file):
    target_img = cv2.imread(target_img_dir + '/' + target_file)
    target_img = img_prep(target_img, 300, 150, True)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(target_img, None)

    ## to show keypoints
    # kp = detector.detect(target_img)
    # dst = cv2.drawKeypoints(target_img, kp, None)
    # cv2.imshow('sample', dst)
    # cv2.waitKey(1000)

    best_match = 'failed'
    lowest_ret = 200

    label_img_dir = 'models/CV2'
    for label_file in os.listdir(label_img_dir):
        label_img = cv2.imread(label_img_dir + '/' + label_file)
        label_img = img_prep(label_img, 300, 150, gray_scale=True)
        (label_kp, label_des) = detector.detectAndCompute(label_img, None)

        matches = bf.match(target_des, label_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)

        # ret = CV2.compareHist(target_hist, base_hist, 0)
        # print('{}: {}'.format(label_file, ret))
        if ret < lowest_ret:
            lowest_ret = ret
            best_match = label_file[:-4]
        else:
            pass
    return best_match, lowest_ret


def validate_similarity(target_img_dir):

    results = []
    score = 0
    for target in os.listdir(target_img_dir):

        pred, similarity_score = similarity_search(target_img_dir, target)
        results.append(['similarity', target[:-7], pred])
        if target[:-7] == pred:
            score += 1
        else:
            pass
    pd.DataFrame(results).to_csv('models/summary/model_similarity_res.csv')

    accuracy = score/len(os.listdir(target_img_dir))
    print('the accuracy is ', round(accuracy, 3))


if __name__ == '__main__':
    validate_similarity()