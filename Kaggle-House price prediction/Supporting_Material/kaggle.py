import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

plt.hist(train_data.SalePrice, color='blue')
#plt.show()

correlation_matrix = train_data.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(correlation_matrix, vmin=0.2, vmax=0.8, square=True, cmap='BuPu')
plt.xlabel('The house features in the x axis', fontsize=13)
plt.ylabel('The house features in the y axis', fontsize=13)
#plt.show()

quality_pivot = train_data.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('OverallQual')
plt.ylabel('Median SalePrice')
plt.xticks(rotation=0)
#plt.show()

plt.scatter(x=train_data['GrLivArea'], y=np.log(train_data.SalePrice))
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
plt.scatter(x=train_data['GarageArea'], y=np.log(train_data.SalePrice))
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
#plt.show()
train_data = train_data[train_data['GrLivArea'] < 4500]
train_data = train_data[train_data['GarageArea'] < 1200]

train_count = train_data.shape[0]
all_data = pd.concat(objs=[train_data, test_data], axis=0)

all_data.drop(['GarageYrBlt'], axis=1, inplace=True)
all_data.drop(['GarageCars'], axis=1, inplace=True)
all_data.drop(['Utilities'], axis=1, inplace=True)
all_data.drop(['Street'], axis=1, inplace=True)
all_data.drop(['Id'], axis=1, inplace=True)

print(all_data.columns[all_data.isna().any()].tolist())
all_data['Alley'] = all_data['Alley'].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['PoolQC'] = all_data['PoolQC'].apply(lambda x: 0 if pd.isna(x) else 1)
print(all_data.columns[all_data.isna().any()].tolist())
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# change to categorical
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['GarageArea']
print(all_data.dtypes)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond',
        'HeatingQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'BsmtExposure', 'GarageFinish',
        'LandSlope', 'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'YrSold', 'MoSold', 'Alley')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data = pd.get_dummies(all_data)

train_data, test_data = all_data[:train_count], all_data[train_count:].drop(['SalePrice'], axis=1)

x_trainn = train_data.drop('SalePrice',axis=1)
y_train = train_data['SalePrice']

print("Linear Regression \n")

classifierLinearRegression = LinearRegression(fit_intercept=True, normalize=False,
                                              copy_X=True, n_jobs=1)
#classifierLinearRegression = classifierLinearRegression.fit(x_trainn,y_train)
#pred = classifierLinearRegression.predict(test_data)

#print(pred)

print("svm")

classifierSVR = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                    C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                    verbose=False, max_iter=-1)


#classifierSVR = classifierSVR.fit(x_trainn, y_train)
#pred2 = classifierSVR.predict(test_data)
#cv_pred.append(pred2)

print("svm")
classifierAllFeatures = RandomForestRegressor(n_estimators=750, criterion='mse',
                max_depth=15, min_samples_split=5, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)
classifierAllFeatures.fit(x_trainn, y_train)
pred3 = classifierAllFeatures.predict(test_data)
print(pred3)
data = pd.read_csv("sample_submission.csv")
data['SalePrice'] = pred3
data.to_csv('sample_submission.csv')



