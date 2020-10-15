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

df_pca1 = pd.DataFrame(df_features)

for i in range(len(df_pca1.iloc[0,:])):
    df_pca1.iloc[:,i] = (df_pca1.iloc[:,i] - df_pca1.iloc[:,i].mean())/df_pca1.iloc[:,i].std()

from sklearn.decomposition import PCA

pca = PCA(n_components=25)
pca_fit = pca.fit(df_pca1)
df_pca1 = pca.fit_transform(df_pca1)


pca_var = pca_fit.explained_variance_ratio_

df_pca2 = df_pca1[:179,:]
df_pca2_test = df_pca1[179:,:]

#47 componants needed for 95% variance
#%%


print(pca_var.sum())

#%%

df = df_features.iloc[:179,:]
df_test = df_features.iloc[179:,:]
df_labels = df_labels.iloc[:179,:]

#%%

for i in range(len(df.iloc[0,:])):
    df.iloc[:,i] = (df.iloc[:,i] - df.iloc[:,i].mean())/df.iloc[:,i].std()


df_cov = np.cov(df)
cov_eigvals, cov_eigvecs = np.linalg.eig(df_cov)

df_pca = cov_eigvecs[:,:]

#%%
from sklearn.manifold import TSNE

#%%

U, S, V = np.linalg.svd(df)

plt.figure(figsize = (8,4))

plt.bar(range(179),S)
plt.yscale('log')
plt.show()



#%%
state = 50

#%%

#regular data

df_embedded = TSNE(n_components=2, random_state = state).fit_transform(df.iloc[:,:])

plt.figure()
plt.title('Unprocessed training data')
plt.scatter(df_embedded[:,0], df_embedded[:,1], c = df_labels)
plt.xlabel('tSNE componant 1')
plt.ylabel('tSNE componant 2')
plt.grid(alpha = 0.5, linestyle = '--')
plt.show()

#%%

#feature selection via anova

fs = SelectKBest(score_func = f_classif, k = 50)
df_anova = fs.fit_transform(df,df_labels)

df_embedded = TSNE(n_components=2, random_state = state).fit_transform(df_anova[:,:])

plt.figure()
plt.title('Training data with ANOVA feature selection')
plt.scatter(df_embedded[:,0], df_embedded[:,1], c = df_labels)
plt.xlabel('tSNE componant 1')
plt.ylabel('tSNE componant 2')
plt.grid(alpha = 0.5, linestyle = '--')
plt.show()

#%%

#normalised data

df_normed = df.copy()

for i in range(len(df_normed.iloc[0,:])):
    df_normed.iloc[:,i] = (df_normed.iloc[:,i] - df_normed.iloc[:,i].mean())/df_normed.iloc[:,i].std()
    

df_embedded = TSNE(n_components=2).fit_transform(df_normed.iloc[:,:])

plt.figure()
plt.title('Training data with normalisation')
plt.scatter(df_embedded[:,0], df_embedded[:,1], c = df_labels)
plt.xlabel('tSNE componant 1')
plt.ylabel('tSNE componant 2')
plt.grid(alpha = 0.5, linestyle = '--')
plt.show()


#%%

#pca

df_embedded = TSNE(n_components=2, random_state = state).fit_transform(df_pca2[:,:7
                                                         ])

plt.figure()

plt.title('Training data with PCA')
plt.scatter(df_embedded[:,0], df_embedded[:,1], c = df_labels)
plt.xlabel('tSNE componant 1')
plt.ylabel('tSNE componant 2')
plt.grid(alpha = 0.5, linestyle = '--')
plt.show()





#%%
df_embedded = TSNE(n_components=2).fit_transform(cov_eigvecs[:,:7])

plt.figure()
plt.scatter(df_embedded[:,0], df_embedded[:,1], c = df_labels)
plt.show()

