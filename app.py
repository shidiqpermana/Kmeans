from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_clustering():
    df = pd.read_csv('Mall_Customers.csv')
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Elbow plot
    sse = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(2, 11), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.savefig('static/elbow_plot.png')
    plt.close()

    # Silhouette Score plot
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        silhouette_scores.append(silhouette_score(scaled_features, labels))
    plt.figure()
    plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.savefig('static/silhouette_plot.png')
    plt.close()

    # Clustering with k=5
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # PCA Visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_features)
    plt.figure()
    for i in range(5):
        plt.scatter(
            reduced_data[df['Cluster'] == i, 0],
            reduced_data[df['Cluster'] == i, 1],
            label=f'Cluster {i}'
        )
    plt.title('Cluster Result with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig('static/kmeans_clusters.png')
    plt.close()

    return render_template('result.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
