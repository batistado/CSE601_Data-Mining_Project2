import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

class DataSet:
    def __init__(self, file_name, path, conv_threshold, max_iterations, k):
        self.file_name = file_name
        self.path = path
        self.k = k
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.read_file()
        self.process_k_means()
        self.process_GMM()

    def read_file(self):
        rows = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                rows.append(re.split("\t+", line))

        self.rows = np.array(rows, dtype=np.float64)
        self.data = np.copy(self.rows)

        if input("Do you want to enter starting points by row id? (Y/N):").upper() == 'Y':
            self.get_initial_points_by_index()
            return

        self.get_initial_points_by_points()

    def get_initial_points_by_index(self):
        initial_indices = []
        count = 0
        while count < self.k:
            idx = int(input("Enter the initial cluster starting point index:"))
            initial_indices.append(idx)
            count += 1

        data_dict = dict()
        for row in self.data:
            data_dict[row[0]] = row[2:]

        self.initial_points = []

        for idx in initial_indices:
            self.initial_points.append(data_dict[idx])

    def get_initial_points_by_points(self):
        print("Now enter starting points for {} clusters for data file {}".format(self.k, self.file_name))
        self.initial_points = []
        count = 0
        while count < self.k:
            line = input("Enter the initial cluster starting points with {} dimenstions separated by tabs:".format(self.data.shape[1] - 2))
            self.initial_points.append([float(x) for x in re.split("\t+", line)])

            if len(self.initial_points[-1]) != self.data.shape[1] - 2:
                raise Exception("Incorrect dimensions for the starting point.")
            count += 1

        self.initial_points = np.array(self.initial_points, dtype=np.float64)

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

        self.mu = centers

    def process_GMM(self):
        self.reg_sigma = 1e-8 * np.identity(len(self.data[0]) - 2)
        self.sigma = np.zeros((self.k, self.data.shape[1] - 2, self.data.shape[1] - 2), dtype='float')
        for dim in range(len(self.sigma)):
            np.fill_diagonal(self.sigma[dim], 1)
        self.pi = np.ones(self.k) / self.k
        log_likelyhoods = []
        
        for i in range(self.max_iterations):  
            prob_matrix = np.zeros((self.data.shape[0], self.k))
            for mu, sig, pi, idx in zip(self.mu, self.sigma, self.pi, range(self.k)):
                sig += self.reg_sigma
                mn = multivariate_normal(mean=mu, cov=sig, allow_singular=True)
                prob_matrix[:, idx] = pi * mn.pdf(self.data[:, 2:]) / np.sum([pi_c * multivariate_normal(mean=mu_c,cov=cov_c, allow_singular=True).pdf(self.data[:, 2:]) 
                for pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.sigma + self.reg_sigma)], axis=0)

            self.mu, self.sigma, self.pi = [], [], []

            for c in range(len(prob_matrix[0])):
                m_c = np.sum(prob_matrix[:,c], axis=0)
                mu_c = (1/m_c) * np.sum(self.data[:, 2:] * prob_matrix[:, c].reshape(len(self.data), 1), axis=0)
                self.mu.append(mu_c)
                self.sigma.append(((1 / m_c) * np.dot((np.array(prob_matrix[:,c]).reshape(len(self.data), 1)*(self.data[:, 2:]-mu_c)).T,(self.data[:, 2:]-mu_c)))+self.reg_sigma)
                self.pi.append(m_c / np.sum(prob_matrix))

            log_likelyhoods.append(np.log(np.sum([self.k*multivariate_normal(mean=self.mu[i],cov=self.sigma[j], allow_singular=True).pdf(self.data[:, 2:]) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.sigma)))])))

        cluster_dict = dict()

        for i, row in enumerate(prob_matrix):
            max_val = float('-inf')
            cluster = None
            for j, col in enumerate(row):
                if col > max_val:
                    max_val = col
                    cluster = j + 1

            if cluster not in cluster_dict:
                cluster_dict[cluster] = []

            cluster_dict[cluster].append(self.data[i])

        for k, v in cluster_dict.items():
            cluster_dict[k] = np.array(v, dtype=np.float64)

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
            ground_truth_dict[r[0]] = r[1]

        algo_dict = dict()

        for r in self.result:
            algo_dict[r[0]] = r[1]

        print(ground_truth_dict, algo_dict)

        i = 0
        while i < self.rows.shape[0]:
            j = i + 1

            while j < self.rows.shape[0]:
                if ground_truth_dict[self.rows[i][0]] ==  ground_truth_dict[self.rows[j][0]]:
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

    def pca(self, cluster_dict):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(self.rows[:, 2:])
        principle_components_matrix = pca.transform(self.rows[:, 2:])
        df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
        self.plot(df, "GMM: {}".format(self.file_name))

    @staticmethod
    def plot(df, title):
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        lm.fig.suptitle(title)
        plt.show()
        path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Plots'))
        lm.savefig('{}/{}.png'.format(path, title))

    

        
def read_data(conv_threshold, max_iterations, k):
    path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Data'))
    data_sets = []
    for file in os.listdir(path):
        data_sets.append(DataSet(file, os.path.join(path, file), conv_threshold, max_iterations, k))

    return data_sets

def main():
    # try:
    conv_threshold = float(input("Enter the convergence threshold:"))
    max_iterations = int(input("Enter max number of iterations:"))
    k = int(input("Enter number of clusters:"))

    
    print("Now Scanning Data directory..")
    data_sets = read_data(conv_threshold, max_iterations, k)
    # except Exception as ex:
    #     print("Something went wrong. Error: " + str(ex))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    main()