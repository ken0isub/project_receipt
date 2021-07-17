import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split


def make_model_cnn(scratched_training_dir, cnn_model_name):
    """
    make CNN model and save the model in './models/CNN'.
    :param scratched_training_dir:the directory where training images are
    :param cnn_model_name: the name of '.h5' file
    :return: None
    """

    scratched_store_dirs = os.listdir(scratched_training_dir)

    X = []
    y = []
    # stores_list = []

    for dir in scratched_store_dirs:
        # stores_list.append(dir)
        for file in os.listdir(scratched_training_dir + '/' + dir):
            img = cv2.imread(scratched_training_dir + '/' + dir + '/' + file)
            X.append(img)
            y.append(dir)

    X = np.array(X)
    y_dummy = pd.get_dummies(pd.Series(y)).values
    y = np.argmax(y_dummy, axis=1)

    # with open('./models/stores_list.txt', 'w') as f:
    #     for c in stores_list:
    #         print(c, file=f)
    #
    # for i, yi in enumerate(y):
    #     print(i, yi, stores_list[yi])

    # print(X.shape)
    # print(y_dummy.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.2, random_state=0, stratify=y_dummy)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(10, 10), input_shape=X_train[0].shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(10, 10)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(y_dummy.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])


    model.fit(X_train, y_train,
              batch_size=8,
              epochs=3,
              verbose=1,
              validation_data=(X_test, y_test))

    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.summary()
    if not os.path.exists('./models/CNN/'):
        os.mkdir('./models/CNN/')
    model.save('./models/CNN/' + cnn_model_name + '.h5')