import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def dataloader():
    random_state = 42
    dataset_train = pd.read_csv("data/adult_data.csv", header=None)
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income']
    dataset_train.columns = column_names
    dataset_test = pd.read_csv("data/adult_test.csv", header=None)
    dataset_test.columns = column_names

    dataset_train = dataset_train.dropna()
    dataset_test = dataset_test.dropna()

    # Bucketize age, assign a bin number to each age
    dataset_train['age'] = np.digitize(dataset_train['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    dataset_test['age'] = np.digitize(dataset_test['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Drop fnlwgt column
    dataset_train = dataset_train.drop(columns=['fnlwgt'])
    dataset_test = dataset_test.drop(columns=['fnlwgt'])

    dataset_train.replace(to_replace=[' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', ' Black'],
                       value=['White', 'Minority', 'Minority', 'Minority', 'Minority'], inplace=True)

    dataset_test.replace(to_replace=[' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', ' Black'],
                      value=['White', 'Minority', 'Minority', 'Minority', 'Minority'], inplace=True)

    # Replace income with booleans, Space before 50k is crucial
    dataset_train['income'] = (dataset_train['income'] == ' >50K').astype(int)
    dataset_test['income'] = (dataset_test['income'] == ' >50K.').astype(int)

    # makes target and target_test, which are income for bias and gender for debias
    sensitive_features = ['gender', 'race']

    # Split training data into training and validation set, random state can be taken as input
    # dataset_train, dataset_validation = train_test_split(dataset_train, test_size=0.2, random_state=random_state,
    #                                                     stratify=dataset_train[['gender', 'race']])

    target_train = dataset_train[["income", "gender", 'race']].copy()
    target_train.replace([' Male', ' Female'], [1, 0], inplace=True)
    target_train.replace(['White', 'Minority'], [1, 0], inplace=True)

    #target_validation = dataset_validation[["income", "gender", 'race']].copy()
    #target_validation.replace([' Male', ' Female'], [1, 0], inplace=True)
    #target_validation.replace(['White', 'Minority'], [1, 0], inplace=True)

    target_test = dataset_test[["income", "gender", 'race']].copy()
    target_test.replace([' Male', ' Female'], [1, 0], inplace=True)
    target_test.replace(['White', 'Minority'], [1, 0], inplace=True)

    # makes dataset and dataset_test, which is the original data set without gender and income, split in a training and
    # test set
    dataset_train = dataset_train.drop(columns=sensitive_features)
    #dataset_validation = dataset_validation.drop(columns=sensitive_features)
    dataset_test = dataset_test.drop(columns=sensitive_features)

    dataset_train = dataset_train.drop("income", axis=1)
    #dataset_validation = dataset_validation.drop("income", axis=1)
    dataset_test = dataset_test.drop("income", axis=1)

    # numerical columns in dataset
    numvars = ['education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'age']
    # categorical column in dataset
    categorical = dataset_train.columns.difference(numvars)

    preprocessor = make_column_transformer(
        (StandardScaler(), numvars),
        (OneHotEncoder(handle_unknown='ignore'), categorical)
    )

    dataset_train = preprocessor.fit_transform(dataset_train)
    #dataset_validation = preprocessor.transform(dataset_validation)
    dataset_test = preprocessor.transform(dataset_test)

    #return dataset_train, dataset_validation, dataset_test, target_train, target_validation, target_test, numvars, categorical
    return dataset_train, dataset_test, target_train, target_test, numvars, categorical

