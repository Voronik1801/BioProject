import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    nx.draw(G, node_color='r', edge_color='b') 
    plt.show()

def create_only_node():
    G = nx.Graph()
    G.add_nodes_from(range(100, 110))
    H = nx.path_graph(10)
    G.add_node(H)
    return G

def create_edge():
    G = nx.Graph()
    G.add_edge(1, 2, weight=4.7)
    G.add_edges_from([(3, 4), (4, 5)], color="green")
    G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    G[1][2]["weight"] = 4.7
    G.edges[1, 2]["weight"] = 4
    return G

G = create_edge()
draw_graph(G)