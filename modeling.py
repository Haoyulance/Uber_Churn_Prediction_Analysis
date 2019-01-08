import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
plt.style.use('ggplot')
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#%% input train and test set
os.getcwd()
os.chdir('/Users/caden/Desktop/project/data')
train = pd.read_csv('cleaned_train_set.csv')
test = pd.read_csv('cleaned_test_set.csv')
vala,test = train_test_split(test.copy(),test_size = .5 ,random_state = 0)
vala = vala.reset_index()
test = test.reset_index()
cols = train.columns
for i in ['signup_dow_1', 'signup_dow_2', 'signup_dow_3', 'signup_dow_4', 'signup_dow_5', 'signup_dow_6', 'signup_dow_7']:
    cols = cols.drop(i)
train = train[cols]
test = test[cols]
vala = vala[cols]
train['churn'].value_counts()
test['churn'].value_counts()
train.head()
train.info()
train.describe()
test.head()
test.info()
test.describe()

x_train = train.iloc[:, 0:-1]
y_train = train['churn']
x_test = test.iloc[:, 0:-1]
y_test = test['churn']
x_val = vala.iloc[:, 0:-1]
y_val = vala['churn']

#%% define evaluation score
def metrics (y_train, y_test, p_train_pre, p_test_pre, threshold = 0.5 ):
    name = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score']
    me_train = [roc_auc_score(y_train, p_train_pre),
                    accuracy_score(y_train, p_train_pre > threshold),
                    precision_score(y_train, p_train_pre > threshold),
                    recall_score(y_train, p_train_pre > threshold),
                    f1_score(y_train, p_train_pre > threshold)
                    ]
    me_test = [roc_auc_score(y_test, p_test_pre),
                    accuracy_score(y_test, p_test_pre > threshold),
                    precision_score(y_test, p_test_pre > threshold),
                    recall_score(y_test, p_test_pre > threshold),
                    f1_score(y_test, p_test_pre > threshold)
                    ]
    metrics = pd.DataFrame({'metrics': name, 'train': me_train, 'test': me_test,
                            }, columns = ['metrics', 'train', 'test']).set_index('metrics')
    print(metrics)

#%% define roc curve
def roc_curve_plot(y_train, y_test, p_train_pre, p_test_pre, name):
    auc_train = roc_auc_score(y_train, p_train_pre)
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train_pre)
    auc_test = roc_auc_score(y_test, p_test_pre)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test_pre)
    plt.figure()
    lw = 1
    plt.plot(fpr_train, tpr_train, color = 'green', lw = lw, label = 'ROC of Train(AUC = %0.4f)' % auc_train)
    plt.plot(fpr_test, tpr_test, color = 'red', lw = lw, label = 'ROC of Test(AUC = %0.4f)' % auc_test)
    plt.plot([0, 1], [0, 1], color='teal', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Curve of comparison')
    plt.legend(loc="lower right")
    plt.savefig(name)
    #plt.show()

#%% Define modeling
def modeling(clf, x_train, y_train, x_test, y_test, name):
    clf.fit(x_train, y_train)
    p_train_pre = clf.predict_proba(x_train)[:,1]
    p_test_pre = clf.predict_proba(x_test)[:,1]
    metrics(y_train, y_test, p_train_pre, p_test_pre)
    roc_curve_plot(y_train, y_test, p_train_pre, p_test_pre, name)

#%% Define feature importance plot
def feature_im(x, y, name):
    im = pd.DataFrame(list(zip(x, y))).sort_values(by=[1], ascending=False)
    im.columns = ['feature', 'importance']
    ax = im.plot.barh()
    t = np.arange(x_train.shape[1])
    ax.set_yticks(t)
    ax.set_yticklabels(im['feature'])
    plt.savefig(name, pad_inches = 0, bbox_inches = 'tight')
    #plt.show()

#%% logistic regression classifier
# scaling for logistic regression and KNN
num_cols = ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train[num_cols])
x_train_model = pd.DataFrame(scaler.transform(x_train[num_cols]), columns = num_cols)
x_test_model = pd.DataFrame(scaler.transform(x_test[num_cols]), columns = num_cols)
x_val_model = pd.DataFrame(scaler.transform(x_val[num_cols]), columns = num_cols)
x_train_ = x_train.drop(columns = num_cols, axis = 1)
x_test_ = x_test.drop(columns = num_cols, axis = 1)
x_val_ = x_val.drop(columns = num_cols, axis = 1)
x_train_model = x_train_.merge(x_train_model, left_index = True, right_index = True, how = 'left')
x_test_model = x_test_.merge(x_test_model, left_index = True, right_index = True, how = 'left')
x_val_model = x_val_.merge(x_val_model, left_index = True, right_index = True, how = 'left')

# logistic regression
#%% grid search for hyperparameter
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
parameter = {
                'C': [.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                'penalty': ['l1', 'l2']
            }
grid = GridSearchCV(clf, parameter, cv = 10, scoring = make_scorer(roc_auc_score))
grid = grid.fit(x_train_model, y_train)
clf = grid.best_estimator_
modeling(clf, x_train_model, y_train, x_val_model, y_val, 'LR_Roc')
print(grid.best_params_)
#%% Uderstand features importance
feature_im(x_train_model.columns, clf.coef_.flatten(), 'LR_im')

#%% KNN classifier
from sklearn.neighbors import KNeighborsClassifier
parameter = {
                'n_neighbors': list(range(5,45,10)),
                'leaf_size': list(range(5,35, 10))
                }
clf = KNeighborsClassifier()
grid = GridSearchCV(clf, parameter, cv = 10, scoring = make_scorer(roc_auc_score))
grid = grid.fit(x_train_model, y_train)
clf = grid.best_estimator_
modeling(clf, x_train_model, y_train, x_val_model, y_val, 'KNN_Roc')
print(grid.best_params_)

#%% Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
# grid search for hyperparameter
parameter = {
                'n_estimators': list(range(400,600,100)),
                'bootstrap': ['True'],
                'max_features':['auto'],
                'criterion': ['gini'],
                'max_depth': list(range(20,40,10)),
                'min_samples_split':list(range(15,20,4)),
                'min_samples_leaf':list(range(15,50,15)),
                'random_state':[0],
                'n_jobs':[-1]
            }
grid = GridSearchCV(clf, parameter, cv = 10, scoring = make_scorer(roc_auc_score))
grid = grid.fit(x_train, y_train)
clf = grid.best_estimator_
modeling(clf, x_train, y_train, x_val, y_val, 'Random_forest_Roc')
print(grid.best_params_)
#%% feature importance
clf.fit(x_train_model, y_train)
feature_im(x_train.columns, clf.feature_importances_, 'RF_im')

#%% Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
parameter = {
                'n_estimators': list(range(60,400,100)),
                'learning_rate':[.1,.2,.5,1],
                'max_depth': list(range(3,10,2)),
                'random_state':[0],
            }
grid = GridSearchCV(clf, parameter, cv = 2, scoring = make_scorer(roc_auc_score))
grid = grid.fit(x_train, y_train)
clf = grid.best_estimator_
modeling(clf, x_train, y_train, x_val, y_val, 'GB_Roc')
print(grid.best_params_)
#%%
clf.fit(x_train_model, y_train)
feature_im(x_train.columns, clf.feature_importances_, 'GB_im')

#%% Final result
modeling(clf, x_train, y_train, x_test, y_test, 'final_Roc')
clf.fit(x_train_model, y_train)
feature_im(x_train.columns, clf.feature_importances_, 'final_im')
#%% out of sample performance
auc_bar = []
for i in range(1000,10001,1000):
    probability = clf.predict_proba(x_test.iloc[i-1000: i])[:,1]
    auc = roc_auc_score(y_test.iloc[i-1000: i], probability)
    auc_bar.append(auc)
bar = np.std(auc_bar)
clf.fit(x_train, y_train)
probability = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probability)
print('The out of sample performance is:')
print('upper bound = ', auc+bar)
print('lower bound = ', auc-bar)
