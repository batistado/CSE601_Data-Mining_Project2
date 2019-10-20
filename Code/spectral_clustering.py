import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import e
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self, file_name, path, k, sigma, max_iterations = 500):
        self.file_name = file_name
        self.path = path
        self.k = k
        self.max_iterations = max_iterations
        self.read_file()
        self.sigma = sigma
        self.process_spectral_clustering()

    def read_file(self):
        rows = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                rows.append(re.split("\t+", line))

        self.rows = np.array(rows, dtype=np.float64)
        self.data = np.copy(self.rows)

        print("Now enter starting points for {} clusters for data file {}".format(self.k, self.file_name))
        self.initial_indices = []
        count = 0
        while count < self.k:
            idx = int(input("Enter the initial cluster starting point index:"))
            self.initial_indices.append(idx)
            count += 1

    def assign_clusters(self):
        cluster_dict = dict()

        for row in self.data:
            min_dist = float("inf")
            min_cluster = -1

            for i, center in enumerate(self.initial_points):
                distance = abs(np.linalg.norm(row[2:] - center))
                if distance < min_dist:
                    min_dist = distance
                    min_cluster = i + 1

            if min_cluster not in cluster_dict:
                cluster_dict[min_cluster] = []

            cluster_dict[min_cluster].append(row)

        for k, v in cluster_dict.items():
            cluster_dict[k] = np.array(v, dtype=np.float64)

        return cluster_dict

    def adjust_centers(self, cluster_dict):
        centers = [None] * self.k
        for k, v in cluster_dict.items():
            centers[k - 1] = list(v[:, 2:].mean(axis=0))

        for i, c in enumerate(centers):
            if c is None:
                centers[i] = self.initial_points[i]
        return np.array(centers, dtype=np.float64)


    def process_k_means(self):
        iter = 1

        cluster_dict = self.assign_clusters()
        centers = self.adjust_centers(cluster_dict)

        while not np.array_equal(centers, self.initial_points) and iter < self.max_iterations:
            self.initial_points = centers
            cluster_dict = self.assign_clusters()
            centers = self.adjust_centers(cluster_dict)
            iter += 1

        self.result = None
        for k, v in cluster_dict.items():
            for r in v:
                r[1] = k

            if self.result is not None:
                self.result = np.concatenate((self.result, v[:,0: 2]), axis=0)
            else:
                self.result = v[:,0: 2]

        self.result = self.result[self.result[:,0].argsort()]
        
        self.pca(cluster_dict)


    def pca(self, cluster_dict):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(self.rows[:, 2:])
        principle_components_matrix = pca.transform(self.rows[:, 2:])
        df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
        self.plot(df, "Spectral Clustering: {}".format(self.file_name))

    @staticmethod
    def plot(df, title):
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        lm.fig.suptitle(title)
        plt.show()
        path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Plots'))
        lm.savefig('{}/{}.png'.format(path, title))

    def process_spectral_clustering(self):
        W = np.zeros((self.rows.shape[0], self.rows.shape[0]), dtype=np.float64)
        D = np.zeros((self.rows.shape[0], self.rows.shape[0]), dtype=np.float64)

        i = 0
        while i < self.data.shape[0]:
            j = i + 1
            while j < self.data.shape[0]:
                wt = e ** (-1 * (np.linalg.norm(self.data[i][2:] - self.data[j][2:])) ** 2 / self.sigma ** 2)

                W[i][j] = wt
                W[j][i] = wt

                D[i][i] += wt

                j += 1

            i += 1

        L = D - W

        w, v = np.linalg.eig(L)
        indices = w.argsort()[:self.k]
        v = v[:,indices]
        
        self.data = np.concatenate((self.data[:, 0:2], v), axis=1)
        
        data_dict = dict()

        for row in self.data:
            data_dict[row[0]] = row[2:]

        self.initial_points = []

        for idx in self.initial_indices:
            self.initial_points.append(data_dict[idx])

        self.process_k_means()

    

        
def read_data(k, sigma):
    path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Data'))
    data_sets = []
    for file in os.listdir(path):
        data_sets.append(DataSet(file, os.path.join(path, file), k, sigma))

    return data_sets

def main():
    # try:
    # TODO: Add support for max iterations input parameter
    k = int(input("Enter the number of clusters to cluster the datasets into:"))
    sigma = float(input("Enter the sigma value:"))
    
    print("Now Scanning Data directory..")
    data_sets = read_data(k, sigma)
    # except Exception as ex:
    #     print("Something went wrong. Error: " + str(ex))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    main()