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
        print(self.process_E_step())

    def read_file(self):
        rows = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                rows.append(re.split("\t+", line))

        self.rows = np.array(rows, dtype=np.float64)
        self.data = np.copy(self.rows)

        print("Now enter initial parameters for {} clusters for data file {}".format(self.k, self.file_name))
        self.initial_params = []
        count = 0
        while count < self.k:
            line = input("Enter the initial cluster starting points with {} dimenstions separated by commas:".format(self.data.shape[1] - 2))
            self.initial_params.append([float(x) for x in line.split(",")])

            if len(self.initial_params[-1]) != 3:
                raise Exception("Please enter the initial parameters in the order: pi, mu, Sigma")
            count += 1

    def process_E_step(self):
        for i in range(self.max_iterations):               
            r_ik = np.zeros((self.data.shape[0], self.k))
            for param, r in zip(self.initial_params, range(len(r_ik[0]))):
                p, m, co = param
                mn = multivariate_normal(mean=m,cov=co)
                print(multivariate_normal.pdf(self.data[:, 2:], mean=m,cov=co))
                print(p * mn.pdf(self.data[:, 2:]))
                r_ik[:,r] = p*mn.pdf(self.data[:, 2:])/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.data[:, 2:]) for pi_c,mu_c,cov_c in self.initial_params],axis=0)

        return r_ik

    def pca(self, cluster_dict):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(self.rows[:, 2:])
        principle_components_matrix = pca.transform(self.rows[:, 2:])
        df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
        self.plot(df, "K-Means: {}".format(self.file_name))

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