import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

def draw_pairplot_raw(X):
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(X,plot_kws={"s": 7})
    plt.savefig("./results/Figure_1.jpg")

def plot_PC12(PC1,PC2):
    plt.clf()
    plt.rcParams['savefig.dpi'] = 300 
    plt.rcParams['figure.dpi'] = 300 
    plt.figure(figsize=(6,6))
    plt.scatter(PC1,PC2,s=12)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title("Figure 2: Principal Component 1 & 2")
    plt.savefig("./Results/Figure_2.jpg")

def draw_pairplot(X, labels,names,img_name):
    features_dict = {names[i]:X[:,i] for i in range(X.shape[1])}
    features_dict["class"]=labels
    data_pd = pd.DataFrame(data=features_dict)
    plt.rcParams['savefig.dpi'] = 300 
    plt.rcParams['figure.dpi'] = 300 
    sns.pairplot(data_pd,hue="class",plot_kws={"s": 7},palette="Paired",)
    
    plt.savefig("./results/%s"%img_name)

def draw_lith_units(pred, dataY, num_cluster=9):
    plt.clf()
    plt.figure(figsize=(6,12))
    plt.gca().invert_yaxis()
    indexs = [np.where(pred==i) for i in range(num_cluster)]
    pred[indexs[4]]=6
    pred[indexs[6]]=4
    pred[indexs[3]]=2
    pred[indexs[2]]=8
    pred[indexs[8]]=7
    pred[indexs[7]]=0
    pred[indexs[0]]=5
    pred[indexs[5]]=3    
    plt.scatter(pred, dataY,marker='o',facecolor='none',edgecolors='blue')
    start=-0.1
    end=num_cluster-0.9
    lines = [101.16,267.82,326.12,412.42,603.42,797.30,889.88,972.00]
    texts = ['I','II','III','IV','V','VI','VII-VIII','IX-X']
    for line in lines:
        plt.plot([start,end],[line]*2,color='red',linewidth=1)
    for i in range(len(texts)):
        plt.gca().text(7.4,lines[i]-20,texts[i],fontsize=14,verticalalignment='center',horizontalalignment='center')
    plt.xticks(np.array(list(range(num_cluster))))
    plt.yticks(np.arange(0,1200,100))
    plt.xlabel('Cluster No.')
    plt.ylabel('mbsf/m')
    plt.title("Figure 5: U1431 K-means lith. units (cluster = 9)")
    plt.savefig("./Results/Figure_5.jpg")

def draw_geo_units(pred, dataY,num_cluster=5):
    plt.clf()
    plt.figure(figsize=(6,12))
    plt.gca().invert_yaxis()
    indexs = [np.where(pred==i) for i in range(num_cluster)]
    pred[indexs[2]]=0
    pred[indexs[0]]=2
    pred[indexs[4]]=1
    pred[indexs[1]]=4
    plt.scatter(pred, dataY,marker='o',facecolor='none',edgecolors='blue')
    start=-0.1
    end=num_cluster-0.9
    lines = [157.4,298.56,807.28,966.33]
    texts = ['Pleistocene','Pliocene','Late Miocene','Early Miocene']
    for line in lines:
        plt.plot([start,end],[line]*2,color='red',linewidth=1)
    for i in range(len(texts)):
        plt.gca().text(3.2,lines[i]-20,texts[i],fontsize=14,verticalalignment='center',horizontalalignment='center')
    plt.xticks(np.array(list(range(num_cluster))))
    plt.yticks(np.arange(0,1200,100))
    plt.xlabel('Cluster No.')
    plt.ylabel('mbsf/m')
    plt.title("Figure 6: U1431 K-means geo. units (cluster = 5)")
    plt.savefig("./Results/Figure_6.jpg")

def draw_inertia(losses, Ks, ylabel, title, imgname):
    plt.clf()
    plt.figure(figsize=(7,5))
    plt.plot(Ks,losses,marker='x',color='blue')
    plt.xticks(np.arange(0,22,2))
    plt.xlabel('k')
    plt.ylabel(ylabel) 
    plt.title(title)   
    plt.savefig(imgname)

def plot_PC12_cluster(PC12,labels,title,img_name):
    plt.clf()
    k = labels.max()+1
    PC1=PC12[:,0]
    PC2=PC12[:,1]
    indexs=[]
    plt.figure(figsize=(7,5))
    for i in range(k):
        indexs.append(np.where(labels==i))
        plt.scatter(PC1[indexs[i]],PC2[indexs[i]],s=12)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.savefig(img_name)