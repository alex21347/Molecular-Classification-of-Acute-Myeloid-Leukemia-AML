#Assignment 3 - Classification with SVM

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import PCA

#%%

df_features = pd.read_csv('C:/Users/alex/Documents/myPython/RUG/Introduction to Data Science 2020/Assignment 3/featuresFlowCapAnalysis.csv', header = None)
df_labels = pd.read_csv('C:/Users/alex/Documents/myPython/RUG/Introduction to Data Science 2020/Assignment 3/labelsFlowCapAnalysis.csv', header = None)

#%%

df_pca1 = pd.DataFrame(df_features)

for i in range(len(df_pca1.iloc[0,:])):
    df_pca1.iloc[:,i] = (df_pca1.iloc[:,i] - df_pca1.iloc[:,i].mean())/df_pca1.iloc[:,i].std()

pca = PCA(n_components=25)
pca_fit = pca.fit(df_pca1)
df_pca1 = pca.fit_transform(df_pca1)

df_pca2 = df_pca1[:179,:]
df_pca2_test = df_pca1[179:,:]


#%%
from sklearn.model_selection import train_test_split

#%%

y = df_labels
df = df_pca2

X_train, X_test, y_train, y_test = train_test_split(df,y, train_size = 0.75, random_state = 52, stratify = y)

#%%

from sklearn import svm

#%%
model = svm.SVC(C = 10, gamma = 0.001, kernel ='rbf')

model.fit(X_train,y_train)


#%%

from sklearn import metrics

pred = model.predict(X_test)

print('Accuracy:', f1.recall_score(pred,y_test))

#%%

#SVM

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

model = svm.SVC(C = 10, gamma = 0.001, kernel ='rbf').fit(X_train,y_train)

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
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100],
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')

grid_result = grid.fit(X_train, y_train.iloc[:])

#%%

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Best: 0.985498 using {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}    F1

#%%
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(model,X_test,y_test.iloc[:])

#%%

pred_proba = model.predict_proba(df_pca2_test)
#%%
pred = model.predict(df_pca2_test)
pred_2 = pred[pred == 2]
pred3 = pd.DataFrame(pred)
#%%
pred4 = pred3[pred3.iloc[:,0] == 2].index
pred6 = pd.DataFrame(pred_proba)
pred6['Index'] = pred3.index

#make sure the index was never scrambled

team_x_prediction = pred3
team_x_prediction.columns = ['Class']
#%%
plt.figure()


plt.scatter(range(180), pred_proba[:,1])


plt.show()


#%%
param_grid = {'C': [0.1,1, 10, 100],
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}

#%%

scores1 = np.zeros((2,3))


#%%
a = 0
for num in ['rbf', 'poly', 'sigmoid']:
    model = svm.SVC(C = 1,
                    gamma = 0.01,
                    kernel = num).fit(X_train,y_train)
    
    
    # define evaluation procedur
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=a)
    
    scores = cross_val_score(model, X_train, y_train, scoring='recall', cv=cv, n_jobs=-1)
    mean_score = scores.mean()
    
    model = model.fit(X_train,y_train.iloc[:])
    pred = model.predict(X_test)
    
    test_score = metrics.f1_score(pred, y_test)
    
    scores1[0,a] = mean_score
    scores1[1,a] = test_score
    
    a = a + 1

#%%

plt.figure()
plt.title('F1 score vs C  (SVM)')
plt.plot(np.arange(0.01,100,1),scores1[0,:], label = ' Training Data')
plt.plot(np.arange(0.01,100,1),scores1[1,:], label = ' Test Data')
plt.xlabel('C')
plt.ylabel('F1 score')
plt.legend()
plt.show()

#%%

x = ['rbf', 'poly', 'sigmoid']

plt.figure()

plt.bar(x, scores1[0,:], width=0.2, color='b', align='center')
plt.bar(x, scores1[1,:], width=0.2, color='g', align='center')

plt.show()


#%%

labels = ['rbf', 'poly', 'sigmoid']
men_means = scores1[0,:]
women_means = scores1[1,:]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label=' Training Data')
rects2 = ax.bar(x + width/2, women_means, width, label=' Test Data')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Recall score')
ax.set_title('Recall score vs kernel')
ax.set_xlabel('Kernel')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(bottom = 0.9, top = 1)
ax.legend()


fig.tight_layout()

plt.show()

















