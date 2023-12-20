from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from naive import naive_split, train_id, dev_id, test_id, features_naive, naive_test
from smart import smart_split, smart_test, features_smart
import sys 

al = float(sys.argv[1])


#x_train, y_train, x_dev, y_dev = naive_split()
x_train, y_train, x_dev, y_dev = smart_split()


ridge = Ridge(alpha=al)
ridge.fit(x_train, y_train)
y_dev_pred = ridge.predict(x_dev)


#x_test_naive = naive_test()
#y_test_pred_naive = ridge.predict(x_test_naive)

x_test_smart = smart_test()
y_test_pred_smart = ridge.predict(x_test_smart)


train_ids = train_id()
dev_ids = dev_id()
test_ids = test_id()


# produce output
def output(ids, preds):
    with open("ridge_out.csv", 'w') as f:
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


# test on test
#output(test_ids, y_test_pred_naive) 
output(test_ids, y_test_pred_smart) 


print(rmsle(y_dev, y_dev_pred))
