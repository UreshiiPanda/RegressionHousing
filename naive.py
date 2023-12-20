import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# get training data & normalize it
train_data = pd.read_csv("my_train.csv", sep=",")
train_ids = train_data.loc[:,["Id"]]

train_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
train_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
train_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage' ]] = train_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].fillna(0)

y_train = train_data.loc[:,["SalePrice"]]
y_train = np.log1p(y_train.to_numpy())
train_data = train_data.drop(columns=['Id', 'SalePrice'])


# get dev data & normalize it 
dev_data = pd.read_csv("my_dev.csv", sep=",")
dev_ids = dev_data.loc[:,["Id"]]

dev_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
dev_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
dev_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage' ]] = dev_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].fillna(0)

y_dev = dev_data.loc[:,["SalePrice"]]
y_dev = np.log1p(y_dev.to_numpy())
dev_data = dev_data.drop(columns=['Id', 'SalePrice'])


# get test data & normalize it 
test_data = pd.read_csv("test.csv", sep=",")
test_ids = test_data.loc[:,["Id"]]
test_data = test_data.drop(columns=['Id'])

test_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
test_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
test_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage' ]] = test_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].fillna(0)


# setup OneHot
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train_data)    
binary_train_data = encoder.transform(train_data)      
binary_test_data = encoder.transform(test_data)                   
binary_dev_data = encoder.transform(dev_data)                   


def naive_split():
    x_train = []
    for row in binary_train_data:
        x_train.append(row[:7226])
    x_train = np.array(x_train)

    x_dev = []
    for row in binary_dev_data:
        x_dev.append(row[:7226])
    x_dev = np.array(x_dev)

    return x_train, y_train, x_dev, y_dev
 

def train_id():
    id_nested_list = train_ids.values.tolist()
    id_list = [id for sublist in id_nested_list for id in sublist] 
    return id_list


def dev_id():
    dev_id_nested_list = dev_ids.values.tolist()
    dev_id_list = [id for sublist in dev_id_nested_list for id in sublist] 
    return dev_id_list


def test_id():
    test_id_nested_list = test_ids.values.tolist()
    test_id_list = [id for sublist in test_id_nested_list for id in sublist] 
    return test_id_list


def features_naive():
    features = encoder.get_feature_names_out().tolist()
    return features


def naive_test():
    x_test = []
    for row in binary_test_data:
        x_test.append(row[:7226])
    x_test = np.array(x_test)
    return x_test


