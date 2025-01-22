#Label Correcting Algorithms for the Shortest Path Problem

import numpy as np
from collections import deque

# Tensor decomposition results (example factors for illustration)
# Spatial factor (W_x): Relationship between nodes
W_x = np.random.rand(5, 5)  # Replace with actual spatial factor matrix
# Temporal factor (W_t): Impact of time on travel cost
W_t = np.random.rand(24)  # Example for hourly variations
# Contextual factor (W_c): Impact of external conditions (e.g., weather, congestion)
W_c = np.random.rand(3)  # Replace with actual contextual factor array


# Compute dynamic edge cost based on tensor factors
def compute_edge_cost(u, v, time, context):
    """
    Compute the dynamic cost of edge (u, v) at a given time and context.

    Args:
    u, v (int): Nodes in the graph.
    time (int): Current time (hourly index in this example).
    context (int): Context index (e.g., weather condition).

    Returns:
    float: Dynamic edge cost.
    """
    return W_x[u, v] * W_t[time] * W_c[context]


# Graph representation (adjacency list)
graph = {
    0: [(1, 1), (2, 1)],
    1: [(2, 1), (3, 1)],
    2: [(3, 1)],
    3: [(4, 1)],
    4: []
}


# Label-Correcting Shortest Path Algorithm
def tensor_based_lcsp(graph, source, target, start_time, context):
    """
    Find the shortest path from source to target using tensor-based LCSP.

    Args:
    graph (dict): Adjacency list of the graph.
    source (int): Source node.
    target (int): Target node.
    start_time (int): Starting time (hourly index).
    context (int): Context index.

    Returns:
    tuple: (shortest_path, cost)
    """
    num_nodes = len(graph)
    labels = [float('inf')] * num_nodes  # Initialize costs as infinity
    labels[source] = 0  # Cost to reach source is 0
    predecessors = [-1] * num_nodes  # To reconstruct the path

    queue = deque([source])  # Initialize queue with source

    while queue:
        u = queue.popleft()  # Dequeue node

        for v, _ in graph[u]:  # Iterate over neighbors
            current_time = (start_time + labels[u]) % 24  # Update time
            edge_cost = compute_edge_cost(u, v, current_time, context)
            new_cost = labels[u] + edge_cost

            if new_cost < labels[v]:  # Relaxation step
                labels[v] = new_cost
                predecessors[v] = u  # Update predecessor
                if v not in queue:
                    queue.append(v)  # Enqueue if not already in the queue

    # Reconstruct the shortest path
    path = []
    node = target
    while node != -1:
        path.append(node)
        node = predecessors[node]
    path.reverse()

    return path, labels[target]


# Example usage
source = 0
target = 4
start_time = 8  # Starting at 8 AM
context = 1  # Example context index

shortest_path, cost = tensor_based_lcsp(graph, source, target, start_time, context)
print("Shortest Path:", shortest_path)
print("Cost:", cost)