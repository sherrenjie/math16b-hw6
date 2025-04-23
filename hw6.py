import numpy as np

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
    return np.allclose(col_sums, np.ones(cols), atol=1e-8)


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
        for neighbour in np.where(A[current] == 1)[0]:
            # First time we see this neighbour
            if dist[neighbour] is None:
                dist[neighbour] = dist[current] + 1
                count[neighbour] = count[current]
                q.append(neighbour)
            # Found another shortest path of the same length
            elif dist[neighbour] == dist[current] + 1:
                count[neighbour] += count[current]

        # Early exit: once we've popped v, all its shortest paths are counted
        if current == v:
            break

    return count[v]
