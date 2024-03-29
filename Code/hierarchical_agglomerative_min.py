import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self, file_name, path, k):
        self.file_name = file_name
        self.path = path
        self.k = k
        self.read_file()
        self.process_hierarchical_clustering()

    def read_file(self):
        rows = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                rows.append(re.split("\t+", line))

        self.rows = np.array(rows, dtype=np.float64)
        self.data = np.copy(self.rows)

    def reduce_clusters(self, cluster_dict):
        global_min = float("inf")
        clusters = None

        keys = list(cluster_dict.keys())

        i = 0
        while i < len(keys):
            j = i + 1

            while j < len(keys):
                local_min = float("inf")

                for row1 in cluster_dict[keys[i]]:
                    for row2 in cluster_dict[keys[j]]:
                        d = abs(np.linalg.norm(row1[2:] - row2[2:]))
                        if  d < local_min:
                            local_min = d

                if local_min < global_min:
                    global_min = local_min
                    clusters = sorted([keys[i], keys[j]])

                j += 1

            i += 1

        cluster_dict[clusters[0]] = np.concatenate((cluster_dict[clusters[0]], cluster_dict[clusters[1]]), axis=0)
        del cluster_dict[clusters[1]]

        return cluster_dict


    def process_hierarchical_clustering(self):
        count = 1
        cluster_dict = dict()

        for row in self.data:
            if count not in cluster_dict:
                cluster_dict[count] = []

            cluster_dict[count].append(row)
            count += 1

        for k, v in cluster_dict.items():
            cluster_dict[k] = np.array(v, dtype=np.float64)


        while len(cluster_dict) > self.k:
            cluster_dict = self.reduce_clusters(cluster_dict)

        final_dict = dict()

        keys = sorted(list(cluster_dict.keys()))

        count = 1
        for key in keys:
            final_dict[count] = cluster_dict[key]
            count += 1

        cluster_dict = final_dict

        self.result = None
        for k, v in cluster_dict.items():
            for r in v:
                r[1] = k

            if self.result is not None:
                self.result = np.concatenate((self.result, v[:,0: 2]), axis=0)
            else:
                self.result = v[:,0: 2]

        self.result = self.result[self.result[:,0].argsort()]

        self.calculate_coefficients(cluster_dict)
        
        self.pca(cluster_dict)

    def calculate_coefficients(self, algo_result):
        P = np.zeros((self.rows.shape[0], self.rows.shape[0]), dtype=np.float64)
        C = np.zeros((self.rows.shape[0], self.rows.shape[0]), dtype=np.float64)

        ground_truth_dict = dict()

        for r in self.rows:
            ground_truth_dict[r[0]] = r

        algo_dict = dict()

        for r in self.result:
            algo_dict[r[0]] = r[1]

        i = 0
        while i < self.rows.shape[0]:
            j = i + 1

            while j < self.rows.shape[0]:
                if ground_truth_dict[self.rows[i][0]][1] ==  ground_truth_dict[self.rows[j][0]][1]:
                    P[i][j] = 1
                    P[j][i] = 1

                if algo_dict[self.rows[i][0]] == algo_dict[self.rows[j][0]]:
                    C[i][j] = 1
                    C[j][i] = 1

                j += 1

            i += 1

        M00 = M10 = M01 = M11 = 0

        for i in range(self.rows.shape[0]):
            for j in range(self.rows.shape[0]):
                if P[i][j] == 1 and C[i][j] == 1:
                    M11 += 1
                elif P[i][j] == 1 and C[i][j] == 0:
                    M10 += 1
                elif P[i][j] == 0 and C[i][j] == 1:
                    M01 += 1
                else:
                    M00 += 1

        rand_index = (M11 + M00) / (M11 + M00 + M10 + M01)
        jaccard = (M11) / (M11 + M10 + M01)

        print("For file {}:\n RandIndex: {}\n Jaccard Coefficient: {}\n".format(self.file_name, rand_index, jaccard))


    # def pca(self, cluster_dict):
    #     pca = PCA(n_components=2, svd_solver='full')
    #     pca.fit(self.rows[:, 2:])
    #     principle_components_matrix = pca.transform(self.rows[:, 2:])
    #     df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
    #     self.plot(df, "Hierarchical Agglomerative: {}".format(self.file_name))

    def pca(self, cluster_dict):
        if self.data.shape[1] > 4:
            pca = PCA(n_components=2, svd_solver='full')
            pca.fit(self.rows[:, 2:])
            principle_components_matrix = pca.transform(self.rows[:, 2:])
            df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
            self.plot(df, "Hierarchical Agglomerative: {}".format(self.file_name))
        else:
            df = pd.DataFrame(data = np.concatenate((self.data[:, 2:], self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
            self.plot(df, "Hierarchical Agglomerative: {}".format(self.file_name))

    @staticmethod
    def plot(df, title):
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        lm.fig.suptitle(title)
        plt.show()
        path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Plots'))
        lm.savefig('{}/{}.png'.format(path, title))

    

        
def read_data(k):
    path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Data'))
    data_sets = []
    for file in os.listdir(path):
        data_sets.append(DataSet(file, os.path.join(path, file), k))

    return data_sets

def main():
    # try:
    k = int(input("Enter the number of clusters to cluster the datasets into:"))
    
    print("Now Scanning Data directory..")
    data_sets = read_data(k)
    # except Exception as ex:
    #     print("Something went wrong. Error: " + str(ex))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    main()