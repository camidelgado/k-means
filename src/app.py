from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


url='https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df_raw=pd.read_csv(url)

df=df_raw[['MedInc','Latitude','Longitude']]
escalador=StandardScaler()
df_norm=escalador.fit_transform(df)

kmeans = KMeans(n_clusters=2)
kmeans.fit(df_norm)

df2=escalador.inverse_transform(df_norm)

df2=pd.DataFrame(df2,columns=['MedInc','Latitude','Longitude'])

df2['Cluster'] = kmeans.labels_