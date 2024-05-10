import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import dearpygui.dearpygui as dpg
from matplotlib.figure import Figure


def data_imputer(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(data)
    columns = data.columns
    return pd.DataFrame(data=imputer, columns=columns)

def normalisation(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    columns = data.columns
    newdata = scaler.transform(data)
    return pd.DataFrame(data=newdata, columns=columns)


def process_data(data_fit):
    if len(data_fit.index) > 10000:
        data_fit = data_fit.sample(n=10000)
    cat_data = data_fit.select_dtypes([object, 'datetime']).columns
    return pd.get_dummies(data_fit, columns=cat_data.values)

def KMeans_method(n_clusters):
    return KMeans(n_clusters=n_clusters, init='k-means++')


def AgglomerativeClustering_method(n_clusters):
    return AgglomerativeClustering(n_clusters=n_clusters)


def DBSCAN_method():
    return DBSCAN()


def SpectralClustering_method(n_clusters):
    return SpectralClustering(n_clusters=n_clusters)


def GaussianMixture_method(n_clusters):
    return GaussianMixture(n_components=n_clusters)


def MeanShift_method():
    return MeanShift()


def predict_data(model, data):
    return model.fit_predict(data)

def sil_plot(X, y_km):
    from matplotlib import cm
    from sklearn.metrics import silhouette_samples
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.savefig("Results/result.png")
    return silhouette_avg


