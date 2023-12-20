# RegressionHousing

### Housing price predictions via regression written in Python &amp; Bash


<a name="readme-top"></a>


<!-- Regression gif -->
![reghouse](https://github.com/UreshiiPanda/RegressionHousing/assets/39992411/941c2548-ef47-4feb-b08f-2e60ea94536b)




<!-- ABOUT THE PROJECT -->
## About The Project

This program is part of my first Kaggle competition which involves predicting housing prices via regression 
[Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Different regression and feature 
engineering techniques were implemented in order to compare results, including: feature encoding/scaling/imputation, 
nonlinearizing the data, linear/regularized/nonlinear regressions. Resulting RMSLE scores on the test data can be 
submitted to the Kaggle competition as a csv file.

#### Results

When all features are binarized equally, the RMSLE on dev came out to .152, but this was improved with feature engineering, 
regularization, and nonlinearization. When only categorical features are binarized, and the numerical features are imputed, 
dropped, or scaled, the RMSLE decreased to .1278. When the data was treated polynomially, and a LinearRegression was 
swapped with a regularized "Ridge" regression, an RMSLE of .1261 on dev was achieved. A Bash loop was written to help find the
the most ideal alpha parameter for the Ridge in this case. This yielded my best RMSLE score on the test data on Kaggle, which was .1385. 



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This program can be run in a Jupyter Notebook via the steps below.


### Installation / Execution Steps in Jupyter Lab:

1. Clone the repo
   ```sh
      git clone https://github.com/UreshiiPanda/RegressionHousing.git
   ```
2. The files can be run separately for various regression results. Note that comments have
   been included which assist in testing out the results of different feature engineering techniques.
   A Bash script has also been included for optimizing alpha on the Ridge regression
   ```sh
       python3 linreg.py
       python3 ridge.py 3.0
       python3 poly.py
   ```

4. Open the project in a Jupyter Notebook and run the cell in graph.ipynb to plot the results
   against each other
   ```sh
       Jupyter Lab
   ```
  
