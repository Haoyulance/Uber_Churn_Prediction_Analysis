import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly
% matplotlib inline
plt.style.use('ggplot')
plotly.tools.set_credentials_file(username='caden_', api_key='VIWpZuSVpfEUSj1hX6ir')
os.getcwd()
os.chdir('/Users/caden/Desktop/project/data')

#%% Data Overview
df = pd.read_csv('raw_data.csv')
raw = df.copy()
train,test = train_test_split(raw.copy(),test_size = .4 ,random_state = 0)
train.head()
train.info()
train['city'].unique().tolist()
np.shape(train)
train.shape[0]
train.shape[1]
train.describe()
train.isnull().sum()
train.nunique()
train.keys().tolist()

##% Data Manipulation
# numeric variables
for i in ['avg_dist', 'avg_surge', 'surge_pct', 'weekday_pct', 'avg_rating_by_driver', 'avg_rating_of_driver', 'trips_in_first_30_days']:
    train[i].plot.hist(bins = 20, color = 'teal', grid = True, zorder = 10)
    plt.title('data overview: ' + i)
    plt.show()
scatter_matrix(train[['avg_dist', 'trips_in_first_30_days', 'weekday_pct']], alpha = 1, figsize = (8,8), color ='teal', diagonal = 'kde')
plt.show()
# catrgorical variables
for i in ['city', 'phone', 'luxury_car_user']:
    train[i].value_counts().plot.bar(grid = True, zorder = 10)
    plt.title('data overview: ' + i)
    plt.show()
# data imputation and preprocessing
def manipulate(dset):
    # dealing with missing values
    dset['phone'] = dset['phone'].fillna('no_phone')
    dset['avg_rating_by_driver'] = dset['avg_rating_by_driver'].fillna(train['avg_rating_by_driver'].median())
    dset['avg_rating_of_driver'] = dset['avg_rating_of_driver'].fillna(train['avg_rating_of_driver'].median())
    # convert time series variables
    dset['last_trip_date'] = pd.to_datetime(dset['last_trip_date'])
    dset['signup_date'] = pd.to_datetime(dset['signup_date'])
    #creative variable: signup day of week
    dset['signup_dow'] = dset['signup_date'].apply(lambda x: x.dayofweek+1)
    #convert bool to int
    dset['luxury_car_user'] = dset['luxury_car_user'].astype(int)
    #one-hot-encoding for categorical variables
    cat_cols = dset.nunique()[dset.nunique()<8].keys().tolist()
    dset_dummies = pd.get_dummies(dset[cat_cols], columns = cat_cols)
    dset = dset.join(dset_dummies, lsuffix = 'train', rsuffix = 'dummies')
    # define a label/target/outcome
    dset['churn'] = (dset.last_trip_date < pd.to_datetime('2014-06-01'))*1
    dset['active'] = (dset.last_trip_date >= pd.to_datetime('2014-06-01'))*1
    return dset
train = manipulate(train)
test = manipulate(test)
# review of timestamp
train_timestamp = train[['last_trip_date', 'signup_date']].copy()
train_timestamp['count'] = 1
train_timestamp = train_timestamp.set_index('signup_date')
train_timestamp['count'].resample("1D").sum().plot(color = 'teal') #对index进行downsample，间隔为1天来计算相同date的个数
plt.show()
train_timestamp = train_timestamp.set_index('last_trip_date')
train_timestamp['count'].resample('1D').sum().plot(color = 'teal')
plt.show()
#separate churn and active customers
churn = train[train['churn'] == 1]
active = train[train['churn'] == 0]
#separate categorical and numerical variable
cat_cols = ['luxury_car_user', 'phone', 'city', 'signup_dow']
num_cols = [x for x in raw.keys() if x not in cat_cols]
time_cols = ['last_trip_date', 'signup_date']
for i in time_cols:
    num_cols.remove(i)

#%% Exploratory Data Analysis
(train['churn'].value_counts()/len(train)).plot.bar(grid = True, zorder = 10)
plt.title('data percentage: churned vs active customers')
plt.show()
def bar_p(cols):
    plt.figure(figsize = (10,7))
    plt.subplot(121)
    (churn[cols].value_counts()*1/len(churn)).plot.bar(grid = True, zorder = 10)
    plt.title(cols + ' of churned customers(percentage)')
    plt.subplot(122)
    (active[cols].value_counts()*1/len(active)).plot.bar(grid = True, zorder = 10)
    plt.title(cols + ' of active customers(percentage)')
    plt.savefig(cols, pad_inches = 0, bbox_inches = 'tight')
for i in cat_cols:
    bar_p(i)
def hist_active_vs_churn(df, cols):
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8.5,5))
    axes[0].hist(churn[cols].values, color = 'teal')
    axes[0].set_title('churned customers')
    axes[0].set_xlabel(cols)
    axes[0].set_ylabel('counts')
    axes[1].hist(active[cols].values, color = 'teal')
    axes[1].set_title('active customers')
    axes[1].set_xlabel(cols)
    axes[1].set_ylabel('counts')
    fig.tight_layout()
    plt.show()
columns = [u'avg_dist', u'avg_rating_by_driver', u'avg_rating_of_driver', u'avg_surge', u'surge_pct', u'trips_in_first_30_days', u'weekday_pct']
for i in columns:
    hist_active_vs_churn(train, i)
# scatter matrix between numberic variables except for the timestamp(red: active, blue:churned)
colors = ['red' if i else 'blue' for i in train['active']]
scatter_matrix(train[num_cols],
                alpha=0.8, figsize=(16, 16), diagonal='hist', c=colors)
#plt.show()
plt.savefig('scatter_ma')
# correlation heatmap about the numberic variables
corr = train[num_cols].corr()
cmap = sns.diverging_palette(230, 80, as_cmap=True)
plt.figure()
sns.heatmap(corr, cmap = cmap, xticklabels = corr.columns, yticklabels = corr.columns, cbar_kws={"shrink": .5}, annot = True)
plt.savefig('correlation.png', pad_inches = 0, bbox_inches = 'tight')

#%% output the cleaned and splitted data
train.info()
test.info()
#%%
selected_cols = train.columns.copy()
for i in ['city', 'phone', 'luxury_car_user', 'signup_date', 'signup_dow', 'active', 'last_trip_date']:
    selected_cols = selected_cols.drop(i)
train[selected_cols].to_csv('cleaned_train_set.csv', index = False, encoding = 'utf-8')
test[selected_cols].to_csv('cleaned_test_set.csv', index = False, encoding = 'utf-8')
