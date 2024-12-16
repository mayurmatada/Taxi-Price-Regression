# %%

from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import tensorflow as tf
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from math import log, sqrt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import svm
import joblib
import datetime


data_raw = pd.read_csv("./taxi_trip_pricing.csv")

# %%
# Fill data from formula of Price = Minute_rate*Minutes + Base + dist_rate*dist
data_fare_fix = data_raw.copy()
data_fare_fix['Trip_Price'] = data_fare_fix['Trip_Price'].fillna(data_fare_fix['Base_Fare'] + (
    data_fare_fix['Per_Km_Rate']*data_fare_fix['Trip_Distance_km']) + (data_fare_fix['Per_Minute_Rate']*data_fare_fix['Trip_Duration_Minutes']))
data_fare_fix['Base_Fare'] = data_fare_fix['Base_Fare'].fillna(data_fare_fix['Trip_Price'] - (
    data_fare_fix['Per_Km_Rate']*data_fare_fix['Trip_Distance_km']) + (data_fare_fix['Per_Minute_Rate']*data_fare_fix['Trip_Duration_Minutes']))
data_fare_fix['Per_Km_Rate'] = data_fare_fix['Per_Km_Rate'].fillna(
    (data_fare_fix['Trip_Price'] - (data_fare_fix['Per_Minute_Rate']*data_fare_fix['Trip_Duration_Minutes']) - data_fare_fix['Base_Fare'])/data_fare_fix['Trip_Distance_km'])
data_fare_fix['Trip_Distance_km'] = data_fare_fix['Trip_Distance_km'].fillna(
    (data_fare_fix['Trip_Price'] - (data_fare_fix['Per_Minute_Rate']*data_fare_fix['Trip_Duration_Minutes']) - data_fare_fix['Base_Fare'])/data_fare_fix['Per_Km_Rate'])
data_fare_fix['Per_Minute_Rate'] = data_fare_fix['Per_Minute_Rate'].fillna(
    (data_fare_fix['Trip_Price'] - (data_fare_fix['Per_Km_Rate']*data_fare_fix['Trip_Distance_km']) - data_fare_fix['Base_Fare'])/data_fare_fix['Trip_Duration_Minutes'])
data_fare_fix['Trip_Duration_Minutes'] = data_fare_fix['Trip_Duration_Minutes'].fillna(
    (data_fare_fix['Trip_Price'] - (data_fare_fix['Per_Km_Rate']*data_fare_fix['Trip_Distance_km']) - data_fare_fix['Base_Fare'])/data_fare_fix['Per_Minute_Rate'])

# %%
# drop rows which are still Na in the above columns
data_fare_fix = data_fare_fix.dropna(
    subset=['Trip_Price', 'Base_Fare', 'Per_Km_Rate', 'Trip_Distance_km', 'Per_Minute_Rate', 'Trip_Duration_Minutes'])

# %%
# convert time of the day and traffic to numerical
data_seriel_cat_fix = data_fare_fix.copy()
data_seriel_cat_fix['Time_of_Day'] = data_fare_fix['Time_of_Day'].replace(
    ["Morning", "Afternoon", "Evening", "Night"], [1, 2, 3, 4])
data_seriel_cat_fix['Traffic_Conditions'] = data_seriel_cat_fix["Traffic_Conditions"].replace(
    ['Low', 'Medium', 'High'], [1, 2, 3])

# %%
# One Hot encode non Seriel Columns
data_hot_encoded = pd.get_dummies(data_seriel_cat_fix, columns=[
                                  'Day_of_Week', 'Weather'], dtype=float, drop_first=True)


# %%
# Drop Any Remaining Na Values
data_clean = data_hot_encoded.dropna()

# %%
# Drop Base Fare, and rates as the model can cheat with it
data_clean = data_clean.drop(
    ['Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate'], axis=1)

# %%
# Deal With Tail Heavy Distributions
data_clean = data_clean.drop(
    data_clean[data_clean['Trip_Distance_km'] > 56].index)

# %%
# Split Train And Test before scaling
X = data_clean.drop(['Trip_Price'], axis=1).values
y = data_clean['Trip_Price'].values
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
# Use Linear ridge regression
reg = linear_model.Ridge()
param_grid = {"alpha": np.arange(0, 10, 0.1)}

grid_search = GridSearchCV(reg, param_grid, cv=10,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)

# %%
# Check Usability of grid Search
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res.head()

# %%
# Get final RMSE of test of Ridge Regressor
reg_opt = grid_search.best_estimator_
reg_test_pred = reg_opt.predict(X_test)
reg_opt_rmse = mean_squared_error(y_test, reg_test_pred)
sqrt(reg_opt_rmse)

# %%
# Setup and Eval SVM
svm_est = svm.SVR(kernel="rbf")
svm_est.fit(X_train, y_train)
svm_pred = svm_est.predict(X_test)
svm_rmse = mean_squared_error(y_test, svm_pred)
sqrt(svm_rmse)

# %%
# Save models
joblib.dump(svm_est, "./models/SVM.pkl")
joblib.dump(reg_opt, "./models/Ridge.pkl")

# %% [markdown]
# **RIDGE REGRESSOR IS BETTER**

# %% [markdown]
# # START NEURAL NETWORKS

# %%
print(y_train.shape)

# %%
# imports


# %%
# function to help batch the data
def pandas_to_dataset(dataframe, labels, batch_size=32, ):
    dataframe = dataframe.copy()
    # To transform the DataFrame into a key-value pair.
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    ds = ds.batch(batch_size)
    return ds


# %%
# Prep data for neural network
""" batch_size = 38
train_ds = pandas_to_dataset(X_train, y_train, batch_size=batch_size)
test_ds = pandas_to_dataset(X_test, y_test, batch_size=27) """

# %%
# Compile and Fit Model
neural_network = tf.keras.Sequential([layers.Dense(8, activation='relu'),
                                      layers.Dense(8, activation='relu'),
                                      layers.Dense(4, activation='relu'),
                                      layers.Dense(2, activation='relu'),
                                      layers.Dropout(0.1),
                                      layers.Dense(1)])
neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# %%
neural_network.fit(np.array(X_train), np.array(y_train), batch_size=38, validation_data=(
    X_test, y_test), epochs=300, callbacks=[tensorboard_callback])


# %%
neural_network_pred = neural_network.predict(X_test)
neural_network_rmse = mean_squared_error(y_test, neural_network_pred)
neural_network_pred

# %% [markdown]
# # AS THE AMOUNT OF DATA IS LESS NEURAL NETWORKS DO NOT PERFORM AS WELL AS CONVENTIONAL MACHINE LEARNING ALGORITHIMS
