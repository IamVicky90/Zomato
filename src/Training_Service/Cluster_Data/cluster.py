from src.logger import logger
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import os
import pickle


class cluster:
    def __init__(self):
        self.log = logger.log()
        self.path = os.path.join(os.getcwd(), 'models', f'kmeans.sav')

    def create_clusters(self, x):
        wcss = []
        try:
            for i in range(1, 11):
                k = KMeans(n_clusters=i, random_state=42)
                k.fit(x)
                wcss.append(k.inertia_)
            self.log.log_writer(
                'Sucessfully completed the wcss list!', 'Cluster.log')
        except Exception as e:
            self.log.log_writer(
                f'Could not complet the wcss list error: {str(e)}', 'Cluster.log', message_type='ERROR')
        try:
            plt.plot(range(1, 11), wcss)
            plt.title('The elbow method')
            plt.xlabel('No. of clusters')
            plt.ylabel('WCSS')
            plt.savefig('src/Training_Service/wcss_image.png')
            self.log.log_writer(
                'Sucessfully saved the wcss image at src/Training_Service/wcss_image.png', 'Cluster.log')
        except Exception as e:
            self.log.log_writer(
                f'Couldnot save the wcss image at src/Training_Service/wcss_image.png error: {str(e)}', 'Cluster.log', 'ERROR')
        try:
            kn = KneeLocator(range(1, 11), wcss,
                             curve='convex', direction='decreasing')
            n_clusters = kn.knee
            kmean = KMeans(n_clusters=n_clusters, random_state=42)
            self.log.log_writer(
                f'The total number of clusters created in this dataframe is/are {n_clusters}', 'Cluster.log')
        except Exception as e:
            self.log.log_writer(
                f'Error occured while calculating the total number of clusters created in this dataframe is/are error: {str(e)}', 'Cluster.log', 'ERROR')
        cluster = kmean.fit_predict(x)
        x_with_cluster_column = self.return_dataframe_with_cluster_column(
            x, cluster)
        try:
            pickle.dump(kmean, open(self.path, 'wb'))
            self.log.log_writer(
                f'Sucessfully dump the kmeans.sav model in {self.path}', 'Cluster.log')
        except Exception as e:
            self.log.log_writer(
                f'Could not dump the kmeans.sav model in {self.path} error: {str(e)}', 'Cluster.log', 'ERROR')
        return x_with_cluster_column

    def return_dataframe_with_cluster_column(self, x, cluster):
        try:
            x['cluster'] = cluster
            self.log.log_writer(
                f'Sucessfully created the cluster column', 'Cluster.log')
            return x
        except Exception as e:
            self.log.log_writer(
                f'Could not create the cluster column error: {str(e)}', 'Cluster.log', 'Error')

    def predict_clusters(self, x_test):
        loaded_model = pickle.load(open(self.path, 'rb'))
        cluster = loaded_model.predict(x_test)
        x_test_with_cluster_column = self.return_dataframe_with_cluster_column(
            x_test, cluster)
        return x_test_with_cluster_column
