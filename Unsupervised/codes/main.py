import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from preprocess_data import preprocess_data
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from utils import *

#-------Load and preprocess the data-------
df = preprocess_data()
X=df.iloc[:,1:]
dataY = np.array(df.iloc[:,0])
print("Data Shape:",X.shape)

#-------Normalize the features-------
scaler = preprocessing.StandardScaler()
for i in range(X.shape[1]):
    x = np.array(X.iloc[:,i]).reshape(-1,1)
    scaler.fit(X=x)
    X.iloc[:,i] = scaler.fit_transform(x).reshape(-1,)
dataX = np.array(X)

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 


#-------Draw pairplot-------
draw_pairplot_raw(X)


#-------Do PCA-------
pca = PCA(n_components=15)
pca.fit(dataX)
pcaX = pca.transform(dataX)
EV = pca.explained_variance_ratio_
print("Explained Variances of PCA: ")
print(EV)


#-------Plot PC1 and PC2-------
PC1 = pcaX[:,0].reshape(-1,1)
PC2 = pcaX[:,1].reshape(-1,1)
plot_PC12(PC1,PC2)


#-------Do k-means and indicate clusters on pairplot-------
y_pred_9 = KMeans(n_clusters=9,random_state=42).fit_predict(dataX)
draw_pairplot(dataX,y_pred_9,X.columns.values,"Figure_3.jpg")
y_pred_5 = KMeans(n_clusters=5,random_state=42).fit_predict(dataX)
draw_pairplot(dataX,y_pred_5,X.columns.values,"Figure_4.jpg")


#-------Create plots of cluster vs. time (lith. units and geo. units)-------
draw_lith_units(y_pred_9, dataY)
draw_geo_units(y_pred_5, dataY)


#-------Try multiple k and plot inertia & silhouette score on original data to check the best k-------
Ks=list(range(1,21))
losses,silhouette_scores=[],[]
for k in Ks:
    model=KMeans(n_clusters=k,random_state=42).fit(dataX)
    losses.append(model.inertia_)
    if k>1:
        ss = silhouette_score(dataX,model.labels_)
        silhouette_scores.append(ss)
draw_inertia(losses, Ks, ylabel='Inertia/Sum of Squared Distances', title="Figure 7a: Elbow Method For Optimal k", imgname="./Results/Figure_7a.jpg")
draw_inertia(silhouette_scores, Ks[1:], ylabel='Silhouette Score', title="Figure 7b: Silhouette Score of Different K", imgname="./Results/Figure_7b.jpg")


#-------Try multiple k and plot inertia & silhouette score on PC12 to check the best k-------
Ks=list(range(1,21))
PC12=pcaX[:,:2]
losses,silhouette_scores=[],[]
for k in Ks:
    model=KMeans(n_clusters=k,random_state=42).fit(PC12)
    losses.append(model.inertia_)
    if k>1:
        ss = silhouette_score(PC12,model.labels_)
        silhouette_scores.append(ss)
draw_inertia(losses, Ks, ylabel='Inertia/Sum of Squared Distances', title="Figure 8a: Elbow Method For Optimal k (on PC12)", imgname="./Results/Figure_8a.jpg")
draw_inertia(silhouette_scores, Ks[1:], ylabel='Silhouette Score', title="Figure 8b: Silhouette Score of Different K (on PC12)", imgname="./Results/Figure_8b.jpg")

#-------Plot the clusters of different k on PC12-------
y_pred_pc_3 = KMeans(n_clusters=3,random_state=42).fit_predict(PC12)
y_pred_pc_4 = KMeans(n_clusters=4,random_state=42).fit_predict(PC12)
y_pred_pc_5 = KMeans(n_clusters=5,random_state=42).fit_predict(PC12)
y_pred_pc_9 = KMeans(n_clusters=9,random_state=42).fit_predict(PC12)
plot_PC12_cluster(PC12,y_pred_pc_3,'Figure 9a: Plot the Clusters (k=3) on PC 1&2','./results/Figure_9a.jpg')
plot_PC12_cluster(PC12,y_pred_pc_4,'Figure 9b: Plot the Clusters (k=4) on PC 1&2','./results/Figure_9b.jpg')
plot_PC12_cluster(PC12,y_pred_pc_5,'Figure 9c: Plot the Clusters (k=5) on PC 1&2','./results/Figure_9c.jpg')
plot_PC12_cluster(PC12,y_pred_pc_9,'Figure 9d: Plot the Clusters (k=9) on PC 1&2','./results/Figure_9d.jpg')







