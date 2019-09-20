# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:21:45 2019

@author: Shriyash Shende

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

r = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\REAL ESTATE\\train.csv')
r_t = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\REAL ESTATE\\test.csv')
SalePrice = r['SalePrice']
r.drop(['SalePrice'], axis = 1, inplace = True)
S = pd.concat([r, r_t], axis = 0)
S.info()
describe = S.describe(include = 'all')
Null_sum = S.isnull().sum()
Null_variable = {}
#Collecting Variables which is having null values
for i,j in Null_sum.iteritems():
    if (j > 0):
        Null_variable.update({i:j})

#identifying
for i,j in Null_variable.items():
    if((j/1460)*100 > 50):
        print(i)

S.drop(['Alley', 'PoolQC','Fence', 'MiscFeature'], axis = 1, inplace = True)
del Null_variable['Alley']  
del Null_variable['PoolQC'] 
del Null_variable['Fence'] 
del Null_variable['MiscFeature']  
S.drop(['Id'], axis =1, inplace = True)

#Replacing NA values
S['BsmtQual'].fillna('TA', inplace =True)
S['BsmtCond'].fillna('TA', inplace =True)
S['BsmtExposure'].fillna('No', inplace =True)
S['BsmtFinType1'].fillna('Unf', inplace =True)
S['BsmtFinType2'].fillna('Unf', inplace =True)

S['Electrical'].fillna('SBrkr', inplace =True)

S['FireplaceQu'].fillna('Gd', inplace =True)

S['GarageCond'].fillna('TA', inplace = True)
S['GarageFinish'].fillna('Unf', inplace = True)
S['GarageQual'].fillna('TA', inplace = True)
S['GarageType'].fillna('Attchd', inplace =True)
S['GarageYrBlt'].fillna(1980, inplace = True)

S['LotFrontage'].fillna(69, inplace = True)
S['MasVnrArea'].fillna(1980, inplace = True)
S['MasVnrType'].fillna(0, inplace = True)

sns.heatmap(S.isnull(), cbar=False) #No null values
n = S.isnull().sum()
Null_variable = {}
for i,j in n.iteritems():
    if (j > 0):
        Null_variable.update({i:j})
        
S['BsmtFinSF1'].fillna(368, inplace = True)
S['BsmtFinSF2'].fillna(0, inplace = True)
S['BsmtFullBath'].fillna(0, inplace = True)
S['BsmtHalfBath'].fillna(0, inplace = True)
S['BsmtUnfSF'].fillna(467, inplace = True)
S['Exterior1st'].fillna('Vinyl5d', inplace = True)
S['Exterior2nd'].fillna('Vinyl5d', inplace = True)

S['Functional'].fillna('Typ', inplace = True)
S['GarageArea'].fillna(480, inplace = True)
S['GarageCars'].fillna(2, inplace = True)

S['KitchenQual'].fillna('TA', inplace = True)
S['MSZoning'].fillna('RL', inplace = True)
S['SaleType'].fillna('WD', inplace = True)
S['TotalBsmtSF'].fillna(989, inplace = True)
S['Utilities'].fillna('AllPub', inplace = True)
sns.heatmap(S.isnull(), cbar=False)



'''
Feature Engineering

Feature  Encoding
MSZoning  SaleType SaleCondition GarageQual  GarageCond PavedDrive  GarageFinish 
FireplaceQu  
GarageType 
Functional 
KitchenQual 
Heating
HeatingQC 
CentralAir 
Electrical
BsmtFinType2  
ExterQual        
ExterCond Foundation  
BsmtQual 
BsmtCond BsmtExposure  
BsmtFinType1 
RoofStyle        1460 non-null object
RoofMatl         1460 non-null object
Exterior1st      1460 non-null object
Exterior2nd      1460 non-null object
MasVnrType       1460 non-null object
Street           1460 non-null object
LotShape         1460 non-null object
LandContour      1460 non-null object
Utilities        1460 non-null object
LotConfig        1460 non-null object
LandSlope        1460 non-null object
Neighborhood     1460 non-null object
Condition1       1460 non-null object
Condition2       1460 non-null object
BldgType         1460 non-null object
HouseStyle       1460 non-null object

Dummies Category
Mszoning - dummies
Street
Lotshape
Utilities
LotConfig
Neighborhood
Condition1,condiction2,BldgType,HouseStyle,RoofStyle,RoofMatl,Exteriot1st,Exterior2nd
MasVnrType, Foundation, BsmtExposure, BsmtFinType1, BsmtFinType2,Heating,CentralAir, Electrical,
GarageType, GarageFinish, PavedDrive, SaleType, 

Labelendoing 

LandContour
LandSlope
ExterQual
ExterCond
BsmtQual
BsmtCond
HeatingQC
KitchenQual
Functional
FireplaceQu
GarageQual
GarageCond
SaleCondition
'''
##Labelencoding

from sklearn.preprocessing import LabelEncoder
label_feature = \
    ('LandContour', 'LandSlope','ExterQual', 'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','Functional'
,'FireplaceQu','GarageQual','GarageCond','SaleCondition')
for i in label_feature:
     le = LabelEncoder() 
     le.fit(list(S[i].values))
     S[i] = le.transform(list(S[i].values))
     
##Reamin variables converted into dummies
S =  pd.get_dummies(S, drop_first = True)



#Spliting Train data and Test data
train = S.loc[:1459]
test = S.loc[1459:]
test  = test.drop(test.index[0])
SalePrice = pd.DataFrame(SalePrice)
Final_train = train.join(SalePrice)



#Train and test
X = Final_train.drop(['SalePrice'], axis = 1)
Y = Final_train['SalePrice']



#using heat map
corrmat = Final_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(Final_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#MODEL 
'''
XGBRegressor
KerasRegressor
Rigid, ElasticNet, Lasso,
'''
import xgboost
regressor=xgboost.XGBRegressor()

booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,Y_train)
random_cv.best_estimator_

regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=2, min_child_weight=4, missing=None, n_estimators=900,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
regressor.fit(X_train, Y_train)
Y_pred_train  = regressor.predict(X_train)
from sklearn.metrics import r2_score
r2_score(Y_train, Y_pred_train, multioutput='variance_weighted') 

Y_pred_test = regressor.predict(X_test)
r2_score(Y_test, Y_pred_test, multioutput='variance_weighted') 

pred = regressor.predict(test)
pL = pd.DataFrame(pred)
pL.rename(columns={0:'SalePrice'}, inplace = True)
g = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\REAL ESTATE\\sample_submission.csv')
result = pd.concat([g['Id'], pL], axis=1)
result.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\REAL ESTATE\\Submission.csv')



from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor()

n_estimators = [100, 500, 900, 1100, 1500]
learning_rate=[0.05,0.1,0.15,0.20]
hyperparameter_grid = {
    'n_estimators':n_estimators,
    'learning_rate':learning_rate,
    }
random_cv1 = RandomizedSearchCV(estimator=regr,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv1.fit(X_train,Y_train)
random_cv1.best_estimator_
regr = AdaBoostRegressor(learning_rate=0.2, loss='linear',n_estimators=500)
regr.fit(X_train, Y_train)
Y_pred_train  = regr.predict(X_train)
r2_score(Y_train, Y_pred_train, multioutput='variance_weighted') 
Y_pred_test = regr.predict(X_test)
r2_score(Y_test, Y_pred_test, multioutput='variance_weighted') 
