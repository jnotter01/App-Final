# Final Project: The Al-GO-rithms Delivery Challenge
# Author: John Notter
# Date: 04/27/2025
# Description: Implements three algorithms for delivery optimization:
# 1. Shortest path between two locations (Dijkstra's Algorithm)
# 2. Minimum Spanning Tree from the hub (Prim's Algorithm)
# 3. Dynamic updates to the graph and recalculating MST

import heapq

def algorithm_1(graph, start, end):
    """
    Algorithm 1: Find the lowest cost path between two locations using Dijkstra's Algorithm.
    Inputs: graph (dict), start node (str), end node (str)
    Returns: (path list, total cost)
    """
    # Priority queue: (cost so far, current node)
    min_heap = [(0, start)]
    # Distance dictionary to keep track of minimum cost to reach each node
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    # To reconstruct the path taken
    previous_nodes = {}

    while min_heap:
        current_cost, current_node = heapq.heappop(min_heap)

        # If we reached the destination node, reconstruct path
        if current_node == end:
            path = []
            while current_node in previous_nodes:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, start)
            return path, current_cost

        for neighbor, weight in graph[current_node]:
            new_cost = current_cost + weight
            if new_cost < distance[neighbor]:
                distance[neighbor] = new_cost
                previous_nodes[neighbor] = current_node
                heapq.heappush(min_heap, (new_cost, neighbor))

    return None, float('inf')  # No path found


def algorithm_2(graph, start):
    """
    Algorithm 2: Build a Minimum Spanning Tree (MST) using Prim's Algorithm.
    Inputs: graph (dict), start node (str)
    Returns: (list of edges in MST, total cost)
    """
    min_heap = [(0, start, None)]  # (cost, current node, parent)
    visited = set()
    mst = []
    total_cost = 0

    while min_heap:
        cost, node, parent = heapq.heappop(min_heap)

        if node in visited:
            continue

        visited.add(node)
        if parent is not None:
            mst.append((parent, node, cost))
            total_cost += cost

        for neighbor, weight in graph[node]: #nice
            if neighbor not in visited:
                heapq.heappush(min_heap, (weight, neighbor, node))

    return mst, total_cost


def algorithm_3(graph, hub, edges_to_remove, edges_to_add):
    """
    Algorithm 3: Dynamically update the graph by removing and adding edges, then rebuild MST.
    Inputs: graph (dict), hub node (str), edges_to_remove (list of str), edges_to_add (list of tuples)
    Returns: (updated MST, total cost)
    """
    # Remove specified edges
    for edge in edges_to_remove:
        node1, node2 = edge.split("-")
        graph[node1] = [(n, w) for (n, w) in graph[node1] if n != node2]
        graph[node2] = [(n, w) for (n, w) in graph[node2] if n != node1]

    # Add new edges
    for node1, node2, weight in edges_to_add:
        graph[node1].append((node2, weight))
        graph[node2].append((node1, weight))

    # Reuse Prim's Algorithm to get updated MST
    return algorithm_2(graph, hub)


# Example usage (based on the provided warehouse graph):
if __name__ == "__main__":
    example_graph = {
        "Warehouse": [("House B", 3), ("House A", 17)],
        "House B": [("Warehouse", 3), ("House A", 5), ("House D", 5), ("House C", 10)],
        "House A": [("Warehouse", 17), ("House B", 5), ("House C", 7)],
        "House C": [("House A", 7), ("House B", 10), ("House D", 15)],
        "House D": [("House B", 5), ("House C", 15)]
    }

    # Test Algorithm 1
    path, cost = algorithm_1(example_graph, "Warehouse", "House D")
    print("Algorithm 1 - Shortest Path from Warehouse to House D:", path, "Cost:", cost)

    # Test Algorithm 2
    mst, total = algorithm_2(example_graph, "Warehouse")
    print("\nAlgorithm 2 - Minimum Spanning Tree:", mst, "Total Cost:", total)

    # Test Algorithm 3
    updated_mst, updated_total = algorithm_3(
        example_graph,
        "Warehouse",
        ["House B-House D"],
        [("House A", "House D", 6)]
    )
    print("\nAlgorithm 3 - Updated MST after changes:", updated_mst, "Updated Total Cost:", updated_total)