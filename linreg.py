from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from naive import naive_split, train_id, dev_id, test_id, features_naive, naive_test
from smart import smart_split, smart_test, features_smart
from statistics import median, mean



#x_train, y_train, x_dev, y_dev = naive_split()
x_train, y_train, x_dev, y_dev = smart_split()

reg = LinearRegression().fit(x_train, y_train)
y_dev_pred = reg.predict(x_dev)

#x_test_naive = naive_test()
#y_test_pred_naive = reg.predict(x_test_naive)

x_test_smart = smart_test()
y_test_pred_smart = reg.predict(x_test_smart)



train_ids = train_id()
dev_ids = dev_id()
test_ids = test_id()


# produce output
def output(ids, preds):
    print("Id,SalePrice")
    for id, price in zip(ids, preds):
        print(f"{id},{np.expm1(price[0])}")


# calc RMSLE (log already applied)
def rmsle(actual, pred):
    # log_actual, log_pred = np.log1p(actual), np.log1p(pred)
    squared_errors = (actual - pred) ** 2
    mean_squared_errors = np.mean(squared_errors)
    rmsle_res = np.sqrt(mean_squared_errors)
    return rmsle_res


# get top 10 pos/neg features
def feats_coeffs(features):
    coeffs = reg.coef_
    fc = dict(zip(features, coeffs.tolist()[0]))
    sorted_fc = sorted(fc.items(), key=lambda x:x[1])

    print("Top 10 most positive Features:")
    for f, c in reversed(sorted_fc[-10:]):
        print(f"  {f} : {c}")
    print()
    print("Top 10 most negative Features:")
    for f, c in sorted_fc[:10]:
        print(f"  {f} : {c}")
 

#test on dev
#print()
#print("Dev Results: ")
#output(dev_ids, y_dev_pred)
#print()


# test on test
#output(test_ids, y_test_pred_naive) 
output(test_ids, y_test_pred_smart) 
#print()


print()
print("RMSLE: ", rmsle(y_dev, y_dev_pred))
print()


# get top 10 features
#feats = features_naive()
#feats = features_smart()
#feats_coeffs(feats)


# get bias term
#print()
#print("Bias Term: ", reg.intercept_[0], '\n')
#print("Bias Term: ", np.expm1(reg.intercept_[0]), '\n')


# get weights
#weights = reg.coef_
#feats = features_smart()
#print("Weights: \n")
#for feature, weight in zip(feats, weights[0]):
#    print(f"{feature}: {weight}")
#print()


# get mean/median
#flat_list = [item for sublist in y_train.tolist() for item in sublist]
#print(np.expm1(median(sorted(flat_list))))
#print(np.expm1(mean(sorted(flat_list))))


