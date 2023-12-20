import pandas as pd


alpha_data = pd.read_csv("ridge_res.csv", sep=",")

# Sort the DataFrame by res
sorted_data = alpha_data.sort_values(by='res')

# Get the 5 rows with the smallest RMSLE's
five_smallest_res = sorted_data.head(5)
print(five_smallest_res)

# Get the corresponding "alpha" value from the row
best_alpha = sorted_data.iloc[0]['alpha']

print(best_alpha)
