import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor,cv


print('1')
print(tf.__version__)
np.set_printoptions(suppress=True)
def tryLabelEncoder(df):
    for column in df.columns:
       if df[column].dtype == type(object):
            le = pp.LabelEncoder()
            df[column] = le.fit_transform(df[column])


# Train Data
train = 'train.csv'
test = 'test.csv'
na  = 'missing'
#'n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale' 
#'penalty','l1_ratio','alpha','max_iter','random_state','n_jobs'
train_data = pd.read_csv(train,na_values='none')
# train_data = pd.get_dummies(train_data)
test_data = pd.read_csv(test,na_values='none')
# test_data = pd.get_dummies(test_data)
print(train_data.head())
# train_pool = Pool(train_data, column_description=CD_FILE)
# test_pool = Pool(test_data, column_description=CD_FILE)
print (train_data.head())
print (test_data.head())
temp = train_data.fillna(na)
# tryLabelEncoder(temp)
# penalty  l1_ratio   alpha
X_train = temp.drop(['time','id'],axis=1).values
Y_train = train_data['time'].values
Y_train = Y_train.ravel()

# Test Data
ID = test_data['id'].values
temp = test_data.fillna(na)
# tryLabelEncoder(temp)
TEST = temp.drop(['id'],axis=1).values
# print(X_train)
# print(Y_train)

# Step0. Pre-processing

# scaler = pp.StandardScaler().fit(X_train)
X = X_train
Y = Y_train

# Spliting Training Data, Training Test Data and Validation Data
X_train,X_tmp,Y_train,Y_tmp = ms.train_test_split(X,Y,test_size = 0.2,random_state =13)
X_test,X_val,Y_test,Y_val = ms.train_test_split(X_tmp,Y_tmp,test_size = 0.2,random_state =13)

# TEST = scaler.transform(TEST)

###############################################################
# Initialize CatBoostRegressor
cat_features = [0]
model = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4)
# Fit model
model.fit(X_train,Y_train,cat_features)
# model.fit(X_train,Y_train)
#model.fit(train_pool, use_best_model=True, eval_set=eval_pool)
# Eval
print(model.get_feature_importance())
predictions = model.predict(X_test)
mse = ((Y_test - predictions) ** 2).mean(axis=0)
print ("MSE : %.4g" % mse)
# Get predictions
pre = model.predict(TEST)
################################################################

# pre = model.predict(TEST)
# Step3. Output file
with open ('output.csv','w') as file:
    file.write('Id,Time\n')
    for i in range(0,len(pre)):
        file.write(str(ID[i]))
        file.write(',')
        if pre[i] < 0:
            file.write('0')
            file.write('\n')
        else:
            file.write(str(pre[i]))
            file.write('\n')


