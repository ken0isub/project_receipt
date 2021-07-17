import numpy as np
import scipy.stats
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score
import os


def ml_data_prep(X, y):
    X_flat = []
    for i in range(X.shape[0]):
        X_flat.append(X[i].flatten())
    X_flat = np.array(X_flat)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


def randomized_search(X_train, X_test, y_train, y_test, model):
    model_param_set_random = {
        'LogisticRegression()': {
            'C': scipy.stats.uniform(10 ** (-5), 10 ** 5),
            'max_iter': [100],
            'penalty': ['l2'],
            'multi_class': ['ovr', 'multinomial'],
            'random_state': [42]
        },
        'LinearSVC()': {
            'C': scipy.stats.uniform(10 ** (-5), 10 ** 5),
            'multi_class': ['ovr', 'crammer_singer'],
            'random_state': [42]
        },
        'SVC()': {
            'C': scipy.stats.uniform(10 ** (-5), 10 ** 5),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'decision_function_shape': ['ovo', 'ovr'],
            'random_state': [42]
        },
        'DecisionTreeClassifier()': {
            'max_depth': scipy.stats.randint(1, 11),
            'random_state': [42]
        },
        'RandomForestClassifier()': {
            'n_estimators': scipy.stats.randint(1, 21),
            'max_depth': scipy.stats.randint(1, 6),
            'random_state': [42]
        },
        'KNeighborsClassifier()': {
            'n_neighbors': scipy.stats.randint(1, 11)
        }
    }

    params = model_param_set_random[str(model)]

    max_score = 0
    best_param = None

    clf = RandomizedSearchCV(model, params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')
    if max_score < score:
        max_score = score
        best_param = clf.best_params_
    print('{} :\n  best_score: {},\n  best_param: {}'.format(str(model), max_score, best_param))
    return clf


def make_ml_model(X_train, X_test, y_train, y_test):
    models_dic = {'logistic_reg': LogisticRegression(),
                  'random_forest': RandomForestClassifier(),
                  'decision_tree': DecisionTreeClassifier(),
                  'svc': SVC(),
                  'knc': KNeighborsClassifier()}

    if not os.path.exists('models/ML'):
        os.mkdir('models/ML')

    for name, model in models_dic.items():
        print('making ' + str(name) + '..')
        clf = randomized_search(X_train, X_test, y_train, y_test, model)
        with open(f'models/ML/model_{name}.pickle', mode='wb') as fp:
            pickle.dump(clf, fp)
