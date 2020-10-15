#Assignment 3 - Classification

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.feature_selection import f_classif, SelectKBest

#%%

df_features = pd.read_csv('C:/Users/alex/Documents/myPython/RUG/Introduction to Data Science 2020/Assignment 3/featuresFlowCapAnalysis.csv', header = None)
df_labels = pd.read_csv('C:/Users/alex/Documents/myPython/RUG/Introduction to Data Science 2020/Assignment 3/labelsFlowCapAnalysis.csv', header = None)

#%%

df_features = df_features.drop(119)
df_labels = df_labels.drop(119)


#%%

df_pca1 = pd.DataFrame(df_features)

for i in range(len(df_pca1.iloc[0,:])):
    df_pca1.iloc[:,i] = (df_pca1.iloc[:,i] - df_pca1.iloc[:,i].mean())/df_pca1.iloc[:,i].std()

from sklearn.decomposition import PCA

pca = PCA(n_components=25)
pca_fit = pca.fit(df_pca1)
df_pca1 = pca.fit_transform(df_pca1)


pca_var = pca_fit.explained_variance_ratio_

df_pca2 = df_pca1[:178,:]
df_pca2_test = df_pca1[178:,:]

#47 componants needed for 95% variance

#%%

df = df_features.iloc[:178,:]
df_test = df_features.iloc[178:,:]
df_labels = df_labels.iloc[:178,:]

#%%

from sklearn.model_selection import train_test_split

#%%

y = df_labels
df = df_pca2

X_train, X_test, y_train, y_test = train_test_split(df,y, train_size = 0.75, random_state = 50, stratify = y)

#%%
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

model = XGBClassifier(eta = 0.01,
                      gamma = 0,
                      max_depth= 3,
                      min_child_weight= 3,
                      n_estimators = 200,
                      scale_pos_weight = 6
                      )
# define evaluation procedur
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=50)

scores = cross_val_score(model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
mean_score = scores.mean()

print(mean_score)
#%%

model = model.fit(X_train,y_train.iloc[:])

pred = model.predict(X_test)


print('Accuracy:', metrics.f1_score(pred,y_test))
#%%
# =============================================================================
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = {
#  'max_depth':[2,3,5,8],
#  'min_child_weight':[2,3,4,5],
#  'scale_pos_weight':[2,5,6,7],
#  'eta':[0.01,0.02,0.04],
#  'n_estimators':[50,200,500,1000],
# }
# 
# 
# 
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
# 
# grid_result = grid.fit(X_train, y_train)
# 
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# 
# =============================================================================

#%%
#Best: 0.965549 using {'eta': 0.01, 'gamma': 0, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 200, 'scale_pos_weight': 6}

################
#Best: 0.972978 using {'eta': 0.01, 'max_depth': 2, 'min_child_weight': 2, 'n_estimators': 1000, 'scale_pos_weight': 5}

#%%

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(model,X_test,y_test.iloc[:])

#%%

pred_proba = model.predict_proba(df_pca2_test)
pred = model.predict(df_pca2_test)
pred_2 = pred[pred == 2]

#%%

pred3 = pd.DataFrame(pred)
pred4 = pred3[pred3.iloc[:,0] == 2].index
pred6 = pd.DataFrame(pred_proba)
pred6['Index'] = pred3.index

team_x_prediction = pred3
team_x_prediction.columns = ['Class']


#%%

team_x_prediction.to_csv('Team_3_Prediction.csv')



#%%
plt.figure()


plt.scatter(range(180), pred_proba[:,1])


plt.show()



























