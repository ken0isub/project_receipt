import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import glob
import shutil
import os


def rename_files(scan_data_path):
    """
    rename filenames to directory name plus serial, then move first file to training dir as a training image
    :param scan_data_path: PATH of the receipt files directory
    :return:
    """
    training_dir = './training'
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    validation_dir = './validation'
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)

    files = os.listdir(scan_data_path)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    else:
        pass

    for store in os.listdir(scan_data_path):
        path = scan_data_path + '/' + store
        files = glob.glob(path + '/*')
        files = [s for s in files if not s.endswith('desktop.ini')]
        file_name = store
        for i, file in enumerate(files):
            os.rename(file, os.path.join(path, file_name + '_{0:02d}.jpg'.format(i)))
            if i == 0:
                shutil.move(os.path.join(path, file_name + '_{0:02d}.jpg'.format(i)), training_dir)
            else:
                shutil.move(os.path.join(path, file_name + '_{0:02d}.jpg'.format(i)), validation_dir)


def img_prep(img, img_width=300, img_top_height=150, gray_scale=True):
    """
    resize the image data to width of 300px and trim top 150px
    :param img: CV2.imread　
    :param img_width: target width, default 300px
    :param img_top_height: target height
    :param gray_scale: True for single channel, False for three channels
    :return: 300 x 150 px image
    """

    if gray_scale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        pass
    img_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    size = (img_width, img_thresh.shape[0]*img_width//img_thresh.shape[1])
    img_thresh_resize = cv2.resize(img_thresh, size)
    img_top = img_thresh_resize[2: img_top_height, 2:size[0]]

    return img_top


def augmentation(img, thr=True, blur=True, erode=True, dilation=True, opening=True, closing=False, mosaic=False):

    methods = [thr, blur, erode, dilation, opening, closing, mosaic]
    filter1 = np.ones((3, 3))
    images = [img]
    augment_func = np.array([
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        lambda x: cv2.erode(x, filter1),
        lambda x: cv2.dilate(x, filter1),
        lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, filter1),
        lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, filter1),
        lambda x: cv2.resize(cv2.resize(x, (x.shape[1] // 5, x.shape[0] // 5)), (x.shape[1], x.shape[0]))
    ])

    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])
    for func in augment_func[methods]:
        images = doubling_images(func, images)

    return images


def run_augmentation(training_data_path, augmented_data_path):
    if not os.path.exists(augmented_data_path):
        os.mkdir(augmented_data_path)
    else:
        pass
    files = os.listdir(training_data_path)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    else:
        pass

    for file in files:
        img = cv2.imread(training_data_path + '/' + file)
        augmented_img = augmentation(img_prep(img))

        if not os.path.exists(augmented_data_path + '/' + file[:-7]):
            os.mkdir(augmented_data_path + '/' + file[:-7])

        for num, im in enumerate(augmented_img):
            cv2.imwrite(augmented_data_path + '/' + file[:-7] + '/' + file[:-4] + '_' + str(num).zfill(2) + '.jpg', im)


def read_images(augmented_data_path):
    augmented_dir = os.listdir(augmented_data_path)

    X = []
    y = []
    stores_list = []

    for dir in augmented_dir:
        stores_list.append(dir)
        for file in os.listdir(augmented_data_path + '/' + dir):
            img = cv2.imread(augmented_data_path + '/' + dir + '/' + file)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img_gray)
            y.append(dir)

    with open('./models/stores_list.txt', 'w') as f:
        for c in stores_list:
            print(c, file=f)

    X = np.array(X)
    le = LabelEncoder()
    y_dummy = le.fit_transform(y)
    y_dummy = np.array(y_dummy)
    # np.save('models/X', X)
    # np.save('models/y_dummy', y_dummy)

    return(X, y_dummy, y)


def prep_cv2(training_dir, cv2_label_dir):
    if not os.path.exists(cv2_label_dir):
        os.mkdir(cv2_label_dir)
    else:
        pass
    files = os.listdir(training_dir)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    else:
        pass

    for num, file in enumerate(files):
        img = cv2.imread(training_dir + '/' + file)
        img = img_prep(img)

        # if not os.path.exists(cv2_label_dir + '/' + file[:-7]):
        #     os.mkdir(cv2_label_dir + '/' + file[:-7])

        cv2.imwrite(cv2_label_dir + '/' + file[:-7] + '.jpg', img)



