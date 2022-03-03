import numpy as np
import networkx as nx
from collections import defaultdict
import sys, copy, time
import scipy.io as sci
import re
import sympy
import math
from gensim import corpora
import gensim
from scipy.sparse import csr_matrix
import hdf5storage
import breadth_first_search as bfs
#from sklearn.preprocessing import normalize
# import collections as cl


def compute_centrality(adj):
    n = len(adj)
    adj = adj + np.eye(n)
    cen = np.zeros(n)
    G = nx.from_numpy_matrix(adj)
    # nodes = betweenness_centrality(G, normalized=True)
    nodes = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-4)
    #nodes = nx.eigenvector_centrality_numpy(G)
    for i in range(len(nodes)):
        cen[i] = nodes[i]

    return cen


def build_multiset(graph_data, maxh, depth):
    prob_map = {}
    graphs = {}
    labels = {}
    alllabels = {}
    num_graphs = len(graph_data[0])

    for gidx in range(num_graphs):
        adj = graph_data[0][gidx]['am']
        graphs[gidx] = adj

    for gidx in range(num_graphs):
        label = graph_data[0][gidx]['nl'].T
        labels[gidx] = label[0]
    
    alllabels[0] = labels


    for deep in range(1, maxh):
        labeledtrees = []
        labels_set = set()
        labels = {}
        labels = alllabels[0]
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            nx_G = nx.from_numpy_matrix(adj)
            label = labels[gidx]
            #print(label)
            
            eigen_cen = compute_centrality(adj)

            for node in range(len(adj)):
                edges = list(bfs.bfs_edges(nx_G, label, eigen_cen, source=node, depth_limit=deep))
                #print(edges)
                bfstree = str(label[node])
                cnt = 0
                for u, v in edges:
                    if cnt < len(list(edges)):
                        bfstree += ','
                    bfstree += str(label[int(v)])
                    cnt += 1
                #print(bfstree)
                labeledtrees.append(bfstree)
                labels_set.add(bfstree)

        labels_set = list(labels_set)
        labels_set = sorted(labels_set)
        index = 0
        labels = {}
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            n = len(adj)
            label = np.zeros(n)

            for node in range(n):
                label[node] = labels_set.index(labeledtrees[node+index])
            index += n

            labels[gidx] = label

        alllabels[deep] = labels


    allPaths = {}
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        nx_G = nx.from_numpy_matrix(adj)
        eigen_cen = compute_centrality(adj)
        paths_graph = []
        judge_set = set()
        label = labels[gidx]
        for node in range(len(adj)):
            
            paths_graph.append(str(node))
            judge_set.add(str(node))
            edges = list(bfs.bfs_edges(nx_G, label, eigen_cen, source=node, depth_limit=depth))
            node_in_path = []
            for u, v in edges:
                node_in_path.append(v)

            pathss = []
            for i in range(len(edges)):
                path = list(nx.shortest_path(nx_G, node, node_in_path[i]))

                strpath = ''
                cnt = 0
                for vertex in path:
                    cnt += 1
                    strpath += str(vertex)
                    if cnt < len(path):
                        strpath += ','

                pathss.append(strpath)

            for path in pathss:
                # print(path)
                vertices = re.split(',', path)
                # print(path)
                rvertices = list(reversed(vertices))

                rpath = ''
                cnt = 0
                for rv in rvertices:
                    cnt += 1
                    rpath += rv
                    if cnt < len(rvertices):
                        rpath += ','

                # print(rpath)

                if rpath not in judge_set:
                    # print(path)
                    judge_set.add(path)
                    paths_graph.append(path)
                else:
                    paths_graph.append(rpath)

        allPaths[gidx] = paths_graph

    PP = {}
    for run in range(maxh):
        labels = alllabels[run]
        #graph_map = {}
        alllabeledpaths = []
        #uniquelabeledpaths = set()
        #multiset_map = {}
        for gidx in range(num_graphs):
            paths = allPaths[gidx]
            #count_map = {}
            tmp_labeledpaths = []
            label = labels[gidx]
            # pprint(label)
            for path in paths:
                labeledpaths = ''
                vertices = re.split(',', path)
                cnt = 0
                for vertex in vertices:
                    cnt += 1
                    labeledpaths += str(int(label[int(vertex)]))
                    if cnt < len(vertices):
                        labeledpaths += ','

                tmp_labeledpaths.append(labeledpaths)

            alllabeledpaths.append(tmp_labeledpaths)

        dictionary = corpora.Dictionary(alllabeledpaths)
        corpus = [dictionary.doc2bow(labeledpaths) for labeledpaths in alllabeledpaths]
        M = gensim.matutils.corpus2csc(corpus)
        PP[run] = M

    return PP


if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = "./kernel"
    # location of the datasets
    DATA_DIR = "../datasets/"
    maxh = 3
    depth = 7 #corresponding to d=6 in the paper
    ds_name = sys.argv[1]
    filename = DATA_DIR + ds_name + '.mat'
    data = sci.loadmat(filename)
    #data = hdf5storage.loadmat(filename)
    # print(type(data['graph']['aff'][0]))
    # print(data['label'])
    graph_data = data['graph']

    print("Dataset: %s\n" % (ds_name))

    start = time.time()

    PP = build_multiset(graph_data, maxh, depth)

    end = time.time()

    print("eclipsed time: %g" % (end - start))

    print("computing kernel matrix...")
    num_graphs = len(graph_data[0])
    K = np.zeros((num_graphs, num_graphs))  # kernel matrix initialization

    for j in range(maxh):
        M = PP[j]
        K += (M.T).dot(M)

    # the following computed kernel can be directly fed to libsvm library
    print("Saving the kernel to the following location: %s/%s_kernel.mat" % (OUTPUT_DIR, ds_name))
    sci.savemat("%s/%s_maxh_%d_maxdepth_%d_kernel.mat" % (OUTPUT_DIR, ds_name, maxh, depth), mdict={'kernel': K})




