from sklearn.pipeline import make_pipeline  
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import numpy as np
from naive import naive_split, train_id, dev_id, test_id, features_naive, naive_test
from smart import smart_split, smart_test, features_smart


#x_train, y_train, x_dev, y_dev = naive_split()
x_train, y_train, x_dev, y_dev = smart_split()
x_test_naive = naive_test()
x_test_smart = smart_test()


# make data non-linear
x_train_poly = np.concatenate([x_train, x_train**2], axis=1)
x_dev_poly = np.concatenate([x_dev, x_dev**2], axis=1)
x_test_naive_poly = np.concatenate([x_test_naive, x_test_naive**2], axis=1)
x_test_smart_poly = np.concatenate([x_test_smart, x_test_smart**2], axis=1)





# setup Models and fit to data
#reg = LinearRegression().fit(x_train_poly, y_train)

ridge = Ridge(alpha=3.0)
ridge.fit(x_train_poly, y_train)

#y_dev_pred = reg.predict(x_dev_poly)
#y_test_pred_smart = reg.predict(x_test_smart_poly)

#x_test_naive = naive_test()
#y_test_pred_naive = reg.predict(x_test_naive_poly)

y_dev_pred = ridge.predict(x_dev_poly)
#y_test_pred_smart = ridge.predict(x_test_smart_poly)





train_ids = train_id()
dev_ids = dev_id()
test_ids = test_id()


# produce output
def output(ids, preds):
    with open("poly_out.csv", 'w') as f:
        f.write("Id,SalePrice\n")
        for id, price in zip(ids, preds):
            f.write(f"{id},{np.expm1(price[0])}\n")



# calc RMSLE (log already applied)
def rmsle(actual, pred):
    # log_actual, log_pred = np.log1p(actual), np.log1p(pred)
    squared_errors = (actual - pred) ** 2
    mean_squared_errors = np.mean(squared_errors)
    rmsle_res = np.sqrt(mean_squared_errors)
    return rmsle_res



#test on dev
#print()
#print("Dev Results: ")
#output(dev_ids, y_dev_pred)
#print()


# test on test
#output(test_ids, y_test_pred_naive) 
#output(test_ids, y_test_pred_smart) 


print(rmsle(y_dev, y_dev_pred))


