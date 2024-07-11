import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

column_name=["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
data=pd.read_csv("auto-mpg.data", names = column_name, na_values = "?", comment= "\t", sep= " ", skipinitialspace= True)

data = data.rename(columns = {"MPG":"target"})

describe = data.describe()

print(data.isna().sum()) #Eksik veri sayısının gösterimi

data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean()) #Eksik verilerin ortalama değer ile doldurulması

sns.displot(data.Horsepower)
plt.show()

corr_matrix = data.corr()

threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

sns.pairplot(data, diag_kind = 'kde', markers = '+')
plt.show()

plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v")
    plt.show()

thr = 2
horsepower_desc = describe["Horsepower"]
q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]

acceleration_desc = describe["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc # q3 - q1
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top= data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] # remove Horsepower outliers

print(data)

plt.figure()
sns.displot(data.target)
plt.show()

(mu,sigma)=norm.fit(data["target"])
print("mu: {}, sigma = {}".format(mu, sigma))

plt.figure()
stats.probplot(data["target"], plot=plt)
plt.show()

data["target"]=np.log1p(data["target"])

plt.figure()
sns.displot(data.target)
plt.show()

(mu,sigma)=norm.fit(data["target"])
print("mu: {}, sigma = {}".format(mu, sigma))

plt.figure()
stats.probplot(data["target"], plot=plt)
plt.show()

skewed_feats=data.apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame(skewed_feats, columns=["skewed"])
print(skewness)

data["Cylinders"]=data["Cylinders"].astype(str)
data["Origin"]=data["Origin"].astype(str)

data=pd.get_dummies(data)

x = data.drop(["target"], axis = 1)
y = data.target

test_size = 0.9
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)

# Standardization
scaler = StandardScaler()  # RobustScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Regression Models

def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R^2: {r2}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
print_metrics(Y_test, y_pred_lr, "Linear Regression")

# Ridge Regression (L2)
ridge = Ridge(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train, Y_train)
ridge = clf.best_estimator_
y_pred_ridge = clf.predict(X_test)
print_metrics(Y_test, y_pred_ridge, "Ridge Regression")

# Lasso Regression
lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)
lasso = clf.best_estimator_
y_pred_lasso = clf.predict(X_test)
print_metrics(Y_test, y_pred_lasso, "Lasso Regression")

# ElasticNet
parametersGrid = {"alpha": alphas, "l1_ratio": np.arange(0.0, 1.0, 0.05)}
eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)
y_pred_enet = clf.predict(X_test)
print_metrics(Y_test, y_pred_enet, "ElasticNet")


# StandardScaler
#     Linear Regression MSE:  0.020632204780133015
#     Ridge MSE:  0.019725338010801216
#     Lasso MSE:  0.017521594770822522
#     ElasticNet MSE:  0.01749609249317252
# RobustScaler:
#     Linear Regression MSE:  0.020984711065869643
#     Ridge MSE:  0.018839299330570554
#     Lasso MSE:  0.016597127172690837
#     ElasticNet MSE:  0.017234676963922273  

#XGBoost

parametersGrid = {
    'nthread': [4],
    'objective': ['reg:linear'],
    'learning_rate': [0.03, 0.05, 0.07],
    'max_depth': [5, 6, 7],
    'min_child_weight': [4],
    'silent': [1],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'n_estimators': [500, 1000]
}
model_xgb = xgb.XGBRegressor()
clf = GridSearchCV(model_xgb, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs=5, verbose=True)
clf.fit(X_train, Y_train)
model_xgb = clf.best_estimator_
y_pred_xgb = clf.predict(X_test)
print_metrics(Y_test, y_pred_xgb, "XGBRegressor")

# Averaged Models
class AveragingModels():
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models=(model_xgb, lasso))
averaged_models.fit(X_train, Y_train)
y_pred_avg = averaged_models.predict(X_test)
print_metrics(Y_test, y_pred_avg, "Averaged Models")

# StandardScaler:
#     Linear Regression MSE:  0.020632204780133015
#     Ridge MSE:  0.019725338010801216
#     Lasso MSE:  0.017521594770822522
#     ElasticNet MSE:  0.01749609249317252
#     XGBRegressor MSE: 0.017167257713690008
#     Averaged Models MSE: 0.016034769734972223
# RobustScaler:
#     Linear Regression MSE:  0.020984711065869643
#     Ridge MSE:  0.018839299330570554
#     Lasso MSE:  0.016597127172690837
#     ElasticNet MSE:  0.017234676963922273
#     XGBRegressor MSE: 0.01753270469361755
#     Averaged Models MSE: 0.0156928574668921
