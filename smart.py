import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


# get training data & normalize it
train_data = pd.read_csv("my_train.csv", sep=",")
train_ids = train_data.loc[:,["Id"]]
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
train_data['MSZoning'] = train_data['MSZoning'].fillna(train_data['MSZoning'].mode()[0])
#train_data = train_data.drop(['Utilities'], axis=1)
#train_data = train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
train_data["Functional"] = train_data["Functional"].fillna("Typ")
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
train_data['KitchenQual'] = train_data['KitchenQual'].fillna(train_data['KitchenQual'].mode()[0])
train_data['Exterior1st'] = train_data['Exterior1st'].fillna(train_data['Exterior1st'].mode()[0])
train_data['Exterior2nd'] = train_data['Exterior2nd'].fillna(train_data['Exterior2nd'].mode()[0])
train_data['SaleType'] = train_data['SaleType'].fillna(train_data['SaleType'].mode()[0])

## transform some numericals into categoricals
train_data['MSSubClass'] = train_data['MSSubClass'].apply(str)
train_data['OverallCond'] = train_data['OverallCond'].astype(str)
train_data['YrSold'] = train_data['YrSold'].astype(str)
train_data['MoSold'] = train_data['MoSold'].astype(str)

# fill in any remaining missing values
train_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
train_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
train_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage' ]] = train_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].fillna(0)

# drop Id's and targets
y_train = train_data.loc[:,["SalePrice"]]
y_train = np.log1p(y_train.to_numpy())
train_data = train_data.drop(columns=['Id', 'SalePrice'])




# get dev data & normalize it 
dev_data = pd.read_csv("my_dev.csv", sep=",")
dev_ids = dev_data.loc[:,["Id"]]
dev_data = dev_data.drop(dev_data[(dev_data['GrLivArea']>4000) & (dev_data['SalePrice']<300000)].index)
dev_data["LotFrontage"] = dev_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
dev_data['MSZoning'] = dev_data['MSZoning'].fillna(dev_data['MSZoning'].mode()[0])
#dev_data = dev_data.drop(['Utilities'], axis=1)
#dev_data = dev_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
dev_data["Functional"] = dev_data["Functional"].fillna("Typ")
dev_data['Electrical'] = dev_data['Electrical'].fillna(dev_data['Electrical'].mode()[0])
dev_data['KitchenQual'] = dev_data['KitchenQual'].fillna(dev_data['KitchenQual'].mode()[0])
dev_data['Exterior1st'] = dev_data['Exterior1st'].fillna(dev_data['Exterior1st'].mode()[0])
dev_data['Exterior2nd'] = dev_data['Exterior2nd'].fillna(dev_data['Exterior2nd'].mode()[0])
dev_data['SaleType'] = dev_data['SaleType'].fillna(dev_data['SaleType'].mode()[0])

## transform some numericals into categoricals
dev_data['MSSubClass'] = dev_data['MSSubClass'].apply(str)
dev_data['OverallCond'] = dev_data['OverallCond'].astype(str)
dev_data['YrSold'] = dev_data['YrSold'].astype(str)
dev_data['MoSold'] = dev_data['MoSold'].astype(str)

# fill in any remaining missing values
dev_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
dev_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
dev_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage' ]] = dev_data[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']].fillna(0)

# drop Id's and targets
y_dev = dev_data.loc[:,["SalePrice"]]
y_dev = np.log1p(y_dev.to_numpy())
dev_data = dev_data.drop(columns=['Id', 'SalePrice'])




# get test data & normalize it 
test_data = pd.read_csv("test.csv", sep=",")
test_ids = test_data.loc[:,["Id"]]
test_data = test_data.drop(columns=['Id'])
test_data["LotFrontage"] = test_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
test_data['MSZoning'] = test_data['MSZoning'].fillna(test_data['MSZoning'].mode()[0])
#test_data = test_data.drop(['Utilities'], axis=1)
#test_data = test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
test_data["Functional"] = test_data["Functional"].fillna("Typ")
test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical'].mode()[0])
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])
test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])
test_data['SaleType'] = test_data['SaleType'].fillna(test_data['SaleType'].mode()[0])

## transform some numericals into categoricals
test_data['MSSubClass'] = test_data['MSSubClass'].apply(str)
test_data['OverallCond'] = test_data['OverallCond'].astype(str)
test_data['YrSold'] = test_data['YrSold'].astype(str)
test_data['MoSold'] = test_data['MoSold'].astype(str)

# fill in any remaining missing values
test_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']] = \
test_data[['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'Electrical', 'MasVnrType', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish', 'FireplaceQu',  'PoolQC', 'MiscFeature', 'Alley', 'Fence']].fillna('None')
test_data[['GarageYrBlt', 'MasVnrArea', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath']] = test_data[['GarageYrBlt', 'MasVnrArea', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath']].fillna(0)





# preprocess the features, apply LabelEncoder only to categorical features
tot_data = pd.concat((train_data, test_data)).reset_index(drop=True)
cols = tot_data.select_dtypes(include=object)
for col in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(tot_data[col].values)) 
    train_data[col] = lbl.transform(list(train_data[col].values))
    dev_data[col] = lbl.transform(list(dev_data[col].values))
    test_data[col] = lbl.transform(list(test_data[col].values))

processed_train_data = train_data.to_numpy()
processed_dev_data = dev_data.to_numpy()
processed_test_data = test_data.to_numpy()



def smart_split():
    train_data = []
    for row in processed_train_data:
        train_data.append(row[:284])
    train_data = np.array(train_data)

    x_dev = []
    for row in processed_dev_data:
        x_dev.append(row[:284])
    x_dev = np.array(x_dev)

    return train_data, y_train, x_dev, y_dev
 

def train_id():
    id_nested_list = train_ids.values.tolist()
    id_list = [id for sublist in id_nested_list for id in sublist] 
    return id_list


def dev_id():
    dev_id_nested_list = dev_ids.values.tolist()
    dev_id_list = [id for sublist in dev_id_nested_list for id in sublist] 
    return dev_id_list


def smart_test():
    x_test = []
    for row in processed_test_data:
        x_test.append(row[:284])
    x_test = np.array(x_test)
    return x_test


def features_smart():
    features = train_data.columns.tolist()
    return features


