import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self, file_name, path, epsilon, min_points):
        self.file_name = file_name
        self.path = path
        self.epsilon = epsilon
        self.min_points = min_points
        self.read_file()
        self.process_dbscan()

    def read_file(self):
        rows = []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                rows.append(re.split("\t+", line))

        self.rows = np.array(rows, dtype=np.float64)
        self.data = np.copy(self.rows)

    def region_query(self, point):
        neighbours = deque()

        for neighbour in self.data:
            if point[0] != neighbour[0]:
                distance = abs(np.linalg.norm(neighbour[2:] - point[2:]))

                if distance < self.epsilon:
                    neighbours.append(neighbour)

        return neighbours

    def expand_cluster(self, cluster_dict, cluster_number, point, neighbours, visited, clustered):
        cluster_dict[cluster_number].append(point)
        clustered.add(point[0])

        while neighbours:
            neighbour = neighbours.popleft()
            if neighbour[0] not in visited:
                visited.add(neighbour[0])
                neighbours2 = self.region_query(neighbour)

                if len(neighbours2) + 1 >= self.min_points:
                    for n in neighbours2:
                        neighbours.append(n)

            if neighbour[0] not in clustered:
                cluster_dict[cluster_number].append(neighbour)
                clustered.add(neighbour[0])


    def process_dbscan(self):
        cluster_number = 0
        cluster_dict = dict()

        visited = set()
        clustered = set()
        noise = []

        for point in self.data:
            if point[0] not in visited:
                visited.add(point[0])

                neighbours = self.region_query(point)

                if len(neighbours) + 1 < self.min_points:
                    clustered.add(point[0])
                    noise.append(point)
                else:
                    cluster_number += 1

                    if cluster_number not in cluster_dict:
                        cluster_dict[cluster_number] = []

                    self.expand_cluster(cluster_dict, cluster_number, point, neighbours, visited, clustered)

        cluster_dict[-1] = noise

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
        
        self.pca(cluster_dict)


    def pca(self, cluster_dict):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(self.rows[:, 2:])
        principle_components_matrix = pca.transform(self.rows[:, 2:])
        df = pd.DataFrame(data = np.concatenate((principle_components_matrix, self.result[:, 1: 2]), axis=1), columns = ['PC1', 'PC2', 'Cluster'])
        self.plot(df, "Density based scan: {}".format(self.file_name))

    @staticmethod
    def plot(df, title):
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        lm.fig.suptitle(title)
        plt.show()
        path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Plots'))
        lm.savefig('{}/{}.png'.format(path, title))

    

        
def read_data(epsilon, min_points):
    path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'Data'))
    data_sets = []
    for file in os.listdir(path):
        data_sets.append(DataSet(file, os.path.join(path, file), epsilon, min_points))

    return data_sets

def main():
    # try:
    epsilon = float(input("Enter the max radius value (epsilon):"))
    min_points = int(input("Enter the min number of points:"))
    
    print("Now Scanning Data directory..")
    data_sets = read_data(epsilon, min_points)
    # except Exception as ex:
    #     print("Something went wrong. Error: " + str(ex))
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    main()