import csv
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt

class GraphStructure():
    def __init__(self):
        self.weights = []
        self.Graphs = []
        self.surv_time = []
        self.X = pd.DataFrame()
        self.list_nodes = ['101.A@OD1', '79.A@NH2', '101.A@OD2', '79.A@NH1', '81.A@O', '74.A@O', '83.A@O',
                            '124.B@OD1', '46.B@NE2', '71.B@NE2', '124.B@OD2', '126.B@N',
                            '54.B@OG1', '52.B@OD1', '52.B@OD2', '54.B@N',
                            '126.A@N', '124.A@OD2', '124.A@OD1', '71.A@NE2', '46.A@NE2',
                            '46.B@NH2', '124.B@OD2', '124.B@OD1', '71.B@NE2', '126.B@N',
                            '125.B@OD2', '71.B@NE2', '124.B@OD1', '126.B@N', '124.B@OD2', '46.B@NE2',
                            '124.B@OD1', '46.B@NE2', '71.B@NE2', '126.B@N', '124.B@OD2',
                            '79.A@NE', '80.A@O', '83.A@N']
    
    def draw_graph(self, G):
        color_map = []
        for node in G:
            if node in self.list_nodes:
                color_map.append('red')
            else: 
                color_map.append('blue')      
        # options = {
        #     'node_color': color_map,
        #     'node_size': 15,
        #     'width': 0.5,
        #     'with_lables':True
        # }
        # nx.draw(G, **options) 
        nx.draw(G, node_color=color_map, with_labels=True, width= 1,node_size = 50, font_size=7)
        # nx.draw(G, cmap = plt.get_cmap('jet'),node_color='red') 
        plt.show()
        # plt.show()

    def calculate_main_values(self, path):
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            dataForAnalys = pd.DataFrame(csv_reader)

        self.stability_hidr_bond = dataForAnalys.loc[:0, 1:] # name of column is stability of the hydrogen bond
        donor_akceptor = dataForAnalys.loc[1:182, :0]
        self.surv_time = list(map(float,dataForAnalys.loc[183:, 1:].values[0]))
        for i in range(1, 73):
            self.weights.append(list(float(w) for w in dataForAnalys.loc[1:182, i:i].values))

        self.donor = [donor[0].split('-')[0] for donor in donor_akceptor.values]
        self.akceptor = [akceptor[0].split('-')[1] for akceptor in donor_akceptor.values]
        self.creat_full_value_graph()
        for i in range(len(self.Graphs)):
            for k in range(len(self.Graphs[i])):
                self.calculate_prop(self.Graphs[i][k], i)

    def load_values_in_graph(self, donor, akceptor, weights):
        Graph = nx.Graph()
        for i in range (len(donor)):
            if(weights[i] != 0.0):
                Graph.add_edge(donor[i], akceptor[i], weight=weights[i])
        return Graph

    def creat_full_value_graph(self):
        n = len(self.weights)
        for i in range(n):
            self.Graphs.append(self.load_values_in_graph(self.donor, self.akceptor, self.weights[i]))
            self.draw_graph(self.Graphs[i])
        for i in range(n):
            self.Graphs[i] = self.uniq_subgraphs(self.Graphs[i])

    def uniq_subgraphs(self, G):
        exsisting_subgraph = []
        graphs = []
        for n in G.nodes:
            nodes_subgraph = list(nx.dfs_edges(G, n))
            sub = nx.classes.function.edge_subgraph(G, nodes_subgraph)
            subgraph = nx.Graph(sub)
            if subgraph not in exsisting_subgraph:
                isomorph = [nx.is_isomorphic(subgraph, exs_subgr) for exs_subgr in exsisting_subgraph]
                if not True in isomorph:
                    new_Graph = nx.Graph()
                    for u, v, w in sub.edges(data=True):
                        new_Graph.add_edge(u, v, weight=w['weight'])
                    graphs.append(new_Graph)
                    exsisting_subgraph.append(subgraph)
        return graphs

    def shortest_path(self, G, source, target):
        def path_cost(G, path):
            return sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
        try:
            x = nx.shortest_simple_paths(G, source,target,weight='weight')
            par = sorted([(path_cost(G,p), p) for p in x])
        except: 
            return 0
        return par[0][0]

    def longest_path(self, G):
        def path_cost(G, path):
            x = [G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
            return sum(x)
        try:
            par = (path_cost(G,nx.dag_longest_path(G)), nx.dag_longest_path(G))
        except:
            return 0
        return par[0]

    def calculate_prop(self, G, i):
        name = ''
        nodes = G.nodes
        for n in nodes:
            name += f'_{n}'
        count_nodes = len(G.nodes)
        columns = self.X.columns

        lable2 = f'sum{name}'
        while lable2 in columns and self.X[lable2][i] != 0:
            lable2 += '_an'

        lable3 = f'short{name}'
        while lable3 in columns and self.X[lable3][i] != 0:
            lable3 += '_an'

        lable4 = f'modularity{name}'
        while lable4 in columns and self.X[lable4][i] != 0:
            lable4 += '_an'

        # Модулярность
        modularity = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
        if lable4 not in self.X.columns:
            self.X[lable4] = np.zeros(72)
        self.X[lable4][i] = modularity


        # Сумма всех путей в графе
        sum = 0
        a = list(G.edges(data=True))
        for j in a:
            sum += j[2]['weight']

        if lable2 not in self.X.columns:
            self.X[lable2] = np.zeros(72)
        self.X[lable2][i] = sum
                
        # Длина самого короткого пути от первой до последней вершины в графе
        dfs = list(nx.dfs_preorder_nodes(G))
        path = 20
        for d in dfs:
            short = self.shortest_path(G, dfs[0], d)
            if short < path and short != 0:
                path = short
        if path == 20:
            path = 0
        if lable3 not in self.X.columns:
            self.X[lable3] = np.zeros(72)
        self.X[lable3][i] = path