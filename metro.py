import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
# Create the random seed number for reproducible results
seedNum = 888
# Set up the number of CPU cores available for multi-thread processing
cpu_num = 6
Xy_original = pd.read_csv("C:/Users/LENOVO/Downloads/Metro_Interstate_Traffic_Volume.csv")
# Take a peek at the dataframe after the import
Xy_original.head(10)
Xy_original.info()
print(Xy_original.isnull().sum())
print('Total number of NaN in the dataframe: ', Xy_original.isnull().sum().sum())
# Convert columns from one data type to another
Xy_original['holiday'] = Xy_original['holiday'].astype('category')
Xy_original['weather_main'] = Xy_original['weather_main'].astype('category')
# Create new columns from the date_time attribute
Xy_original['date_time'] = pd.to_datetime(Xy_original['date_time'])
Xy_original['date_month'] = pd.DatetimeIndex(Xy_original['date_time']).month
Xy_original['date_month'] = Xy_original['date_month'].astype('category')
Xy_original['date_weekday'] = pd.DatetimeIndex(Xy_original['date_time']).weekday
Xy_original['date_weekday'] = Xy_original['date_weekday'].astype('category')
Xy_original['date_hour'] = pd.DatetimeIndex(Xy_original['date_time']).hour
Xy_original['date_hour'] = Xy_original['date_hour'].astype('category')
Xy_original['targetVar'] = Xy_original['traffic_volume']

# Drop the un-needed features
Xy_original.drop(columns=['date_time', 'weather_description', 'traffic_volume'], inplace=True)
# Take a peek at the dataframe after the cleaning
Xy_original.head(10)

Xy_original.info()

print(Xy_original.isnull().sum())
print('Total number of NaN in the dataframe: ', Xy_original.isnull().sum().sum())

# Use variable totCol to hold the number of columns in the dataframe
totCol = len(Xy_original.columns)
# Set up variable totAttr for the total number of attribute columns
totAttr = totCol-1
targetCol = totCol
# Standardize the class column to the name of targetVar if required
# Xy_original = Xy_original.rename(columns={'traffic_volume': 'targetVar'})
if targetCol == totCol:
    X_original = Xy_original.iloc[:,0:totAttr]
    y_original = Xy_original.iloc[:,totAttr]
else:
    X_original = Xy_original.iloc[:,1:totCol]
    y_original = Xy_original.iloc[:,0]

print("Xy_original.shape: {} X_original.shape: {} y_original.shape: {}".format(Xy_original.shape, X_original.shape, y_original.shape))

# Set up the number of row and columns for visualization display. dispRow * dispCol should be >= totAttr
dispCol = 4
if totAttr % dispCol == 0 :
    dispRow = totAttr // dispCol
else :
    dispRow = (totAttr // dispCol) + 1
    
# Set figure width to display the data visualization plots
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = dispCol*4
fig_size[1] = dispRow*4
plt.rcParams["figure.figsize"] = fig_size

# Histograms for each attribute
X_original.hist(layout=(dispRow,dispCol))

# Density plot for each attribute
X_original.plot(kind='density', subplots=True, layout=(dispRow,dispCol))

# Sample code for performing one-hot-encoding before splitting into trainig and test

X_original = pd.get_dummies(X_original)
print(X_original.info())

# Use 75% of the data to train the models and the remaining for testing/validation

testDataset_size = 0.25
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_original, y_original, test_size=testDataset_size, random_state=seedNum)
print("X_train_df.shape: {} y_train_df.shape: {}".format(X_train_df.shape, y_train_df.shape))
print("X_test_df.shape: {} y_test_df.shape: {}".format(X_test_df.shape, y_test_df.shape))

# We finalize the training and testing datasets for the modeling activities
X_train = X_train_df.values
y_train = y_train_df.values
X_test = X_test_df.values
y_test = y_test_df.values
print("X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape))
print("X_test.shape: {} y_test.shape: {}".format(X_test.shape, y_test.shape))

# Set up Algorithms Spot-Checking Array
#startTimeModule = datetime.now()
models = []
models.append(('LR', LinearRegression(n_jobs=cpu_num)))
models.append(('RR', Ridge(random_state=seedNum)))
models.append(('LASSO', Lasso(random_state=seedNum)))
models.append(('EN', ElasticNet(random_state=seedNum)))
models.append(('CART', DecisionTreeRegressor(random_state=seedNum)))
models.append(('KNN', KNeighborsRegressor(n_jobs=cpu_num)))
models.append(('RF', RandomForestRegressor(random_state=seedNum, n_jobs=cpu_num)))
models.append(('ET', ExtraTreesRegressor(random_state=seedNum, n_jobs=cpu_num)))
models.append(('GBM', GradientBoostingRegressor(random_state=seedNum)))
models.append(('XGB', XGBRegressor(random_state=seedNum, n_jobs=cpu_num)))
results = []
names = []
metrics = []

# Set up the comparison array
model = RandomForestRegressor(n_estimators=300, random_state=seedNum, n_jobs=cpu_num)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('RMSE for the model is: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('R2 for the model is: ', r2_score(y_test, predictions))

model = ExtraTreesRegressor(n_estimators=700, random_state=seedNum, n_jobs=cpu_num)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('RMSE for the model is: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('R2 for the model is: ', r2_score(y_test, predictions))