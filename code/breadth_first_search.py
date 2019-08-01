import networkx as nx
import copy
from collections import deque

__all__ = ['bfs_edges', 'bfs_tree', 'bfs_predecessors', 'bfs_successors']


def generic_bfs_edges(G, label, source, neighbors=None, depth_limit=None):
    
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    neigh = list(neighbors(source))
    #print(neigh)
    neighlabel = []
    for nei in neigh:
        neighlabel.append(label[int(nei)])
    #print(neighlabel)
    neighindex = sorted(range(len(neighlabel)), key=lambda k: neighlabel[k])
    sortedneighbor = []
    for ele in neighindex:
        sortedneighbor.append(neigh[ele])

    queue = deque([(source, depth_limit, iter(sortedneighbor))])
    #print(type(neighbors(source)))
    #print(list(iter(sortedneighbor)))

    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    chil = list(neighbors(child))
                    chillabel = []
                    for ch in chil:
                        chillabel.append(label[int(ch)])

                    chilindex = sorted(range(len(chillabel)), key=lambda k: chillabel[k])
                    sortedneighbor = []
                    for ele in chilindex:
                        sortedneighbor.append(chil[ele])
                    
                    queue.append((child, depth_now - 1, iter(sortedneighbor)))
        except StopIteration:
            queue.popleft()


def bfs_edges(G, label, source, reverse=False, depth_limit=None):
    
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    # TODO In Python 3.3+, this should be `yield from ...`
    for e in generic_bfs_edges(G, label, source, successors, depth_limit):
        yield e