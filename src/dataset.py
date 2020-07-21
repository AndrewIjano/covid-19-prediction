import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold


def load(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(' initial number of lines:', len(dataset))
    return dataset


def get_exam_data(dataset):
    for i, exam_name in enumerate(['pcr', 'igm', 'igg']):
        exam_dataset = dataset[dataset[exam_name].notna()]
        print(f'\n## {exam_name} ##')
        print(f'number of lines:', len(exam_dataset))
        exam_dataset = exam_dataset.fillna(exam_dataset.median())
        exam_dataset = exam_dataset.to_numpy()
        np.random.shuffle(exam_dataset)

        X = normalize(exam_dataset[:, 3:])
        y = exam_dataset[:, i:i+1]

        yield exam_name, (X, y)


def normalize(X, normalization_type='min-max'):
    if normalization_type == 'z':
        mean_to_zero = (X - np.mean(X, axis=0))
        stdev = np.std(X, axis=0)
        return mean_to_zero / (1 if (stdev == 0).any() else stdev)

    if normalization_type == 'min-max':
        min_x = X.min(axis=0)
        max_x = X.max(axis=0)
        return (X - min_x) / (max_x - min_x + 1)

    return X


def get_cross_validaion_sets(X, y, k=10):
    kf = KFold(n_splits=10)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, y_train = prepare_dataset(X_train, y_train)
        X_test, y_test = prepare_dataset(X_test, y_test)

        yield (X_train, y_train), (X_test, y_test)


def prepare_dataset(X, y):
    X, y = oversample(X, y)
    X, y = map(numpy_to_tensor, (X, y))
    return X, get_array(y)


def oversample(X, y):
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)
    return (X, y)


def numpy_to_tensor(np):
    return torch.from_numpy(np).float()


def get_array(y, n=1):
    return torch.squeeze(y)
