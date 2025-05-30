import numpy as np
from collections import deque 

def is_stochastic(M):
    """
    Return True if the given matrix M is row-stochastic,
    meaning each row sums to 1.
    """
    if not isinstance(M, np.ndarray) or M.ndim != 2:
        return False

    rows, cols = M.shape
    if rows != cols:                       # square matrix check
        return False

    if np.any(M < 0):                      # non-negativity check
        return False

    col_sums = M.sum(axis=0)               # column sums
    return np.allclose(M.sum(axis=1), 1.0, atol=1e-5)


def adjacency(graph_dict):
    """
    Given a directed graph represented as a dictionary where keys are node labels
    and values are lists of neighbors, return the corresponding adjacency matrix
    as a NumPy array.
    """
    nodes = sorted(graph_dict.keys())
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    A = np.zeros((n, n), dtype=int)

    for u in nodes:
        for v in graph_dict[u]:
            if v in index:                 # ignore unknown neighbours, just in case
                A[index[u], index[v]] = 1

    return A


def count_shortest_paths(A, u, v):
    """
    Count the number of shortest paths from node u to node v in an unweighted graph.
    A is the adjacency matrix of the graph.
    """
    n = A.shape[0]
    if u == v:                     # the trivial path of length 0
        return 1

    # Breadth-first search while counting paths
    dist = [None] * n              # distance from u (None means unseen)
    count = [0] * n                # number of shortest paths to each node

    dist[u] = 0
    count[u] = 1
    q = deque([u])

    while q:
        current = q.popleft()

        # consider every neighbour that has ≥1 edge from `current`
        for neighbour in np.where(A[current] > 0)[0]:
            w = int(A[current, neighbour])          # number of parallel edges

            if dist[neighbour] is None:        # first time we reach neighbour
                dist[neighbour]  = dist[current] + 1
                count[neighbour] = count[current] * w
                q.append(neighbour)

            elif dist[neighbour] == dist[current] + 1:
                count[neighbour] += count[current] * w
                
    return count[v] if dist[v] is not None else 0  # Return 0 if v is unreachable