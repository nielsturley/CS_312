#!/usr/bin/python3
from BinaryHeap import BinaryHeap
from CS312Graph import *
import time


class NetworkRoutingSolver:
    def __init__(self):
        pass

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        self.dest = destIndex

        begin_node = self.network.nodes[self.source]
        end_node = self.network.nodes[self.dest]
        path_edges = []
        total_length = end_node.dist

        # Iterate back through the path to find the edges
        if total_length != float('inf'):
            reached_source = False
            while not reached_source:
                next_node = end_node.path
                path_edges.append((end_node.loc, next_node.loc, '{:.0f}'.format(end_node.dist - next_node.dist)))
                if next_node == begin_node:
                    reached_source = True
                else:
                    end_node = next_node
        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths(self, srcIndex, use_heap=False):
        self.source = srcIndex
        t1 = time.time()

        if use_heap:
            self.heap(srcIndex)
        else:
            self.array(srcIndex)

        t2 = time.time()
        return (t2 - t1)

    """Time complexity: O(V^2). Inserting into the pq takes O(1) time, and we do this V times. Removing from the pq 
        takes O(V) time, and we do this V times. Changing the distance of a node takes O(1) time, and we do this E 
        times. Thus, O( (V*1) + (V*V) + (E*1) ) = O(V^2)."""
    """Space complexity: O(V). We create a pq of size V to store the nodes."""
    def array(self, srcIndex):
        # Initialize the array
        pq = set()
        for node in self.network.nodes:
            node.dist = float('inf')
            node.known = False
            node.path = None
        self.network.nodes[srcIndex].dist = 0
        pq.add(self.network.nodes[srcIndex])

        # Run Dijkstra's algorithm
        while len(pq) > 0:
            # Find the node with the smallest distance
            min_dist = float('inf')
            min_node = None
            for node in pq:
                if not node.known and node.dist < min_dist:
                    min_dist = node.dist
                    min_node = node
            if min_node is None:
                break
            pq.remove(min_node)

            # Update the distances of the neighbors
            for edge in min_node.neighbors:
                if edge.dest.dist > min_node.dist + edge.length:
                    edge.dest.dist = min_node.dist + edge.length
                    edge.dest.path = min_node
                    if edge.dest.known:
                        pq.remove(edge.dest)
                    pq.add(edge.dest)
            min_node.known = True



    """
    THIS ONE WORKS
        def array(self, srcIndex):
        # Initialize the array
        pq = []
        for node in self.network.nodes:
            node.dist = float('inf')
            node.known = False
            node.path = None
        self.network.nodes[srcIndex].dist = 0
        pq.extend(self.network.nodes)

        # Run Dijkstra's algorithm
        while len(pq) > 0:
            # Find the node with the smallest distance
            min_dist = float('inf')
            min_node = pq[0]
            for node in self.network.nodes:
                if not node.known and node.dist < min_dist:
                    min_dist = node.dist
                    min_node = node

            pq.remove(min_node)

            # Update the distances of the neighbors
            for edge in min_node.neighbors:
                if edge.dest.dist > min_node.dist + edge.length:
                    edge.dest.dist = min_node.dist + edge.length
                    edge.dest.path = min_node
            min_node.known = True"""

    """Time complexity: O( (V + E) * log(V)). Inserting into the pq takes O(log(V)) time, and we do this V times. 
        Removing from the pq takes O(log(V)) time, and we do this V times. Changing the distance of a node takes O(log(
        V)) time, and we do this E times. Thus, O( (V*log(V)) + (V*log(V)) + (E*log(V)) ) = O( (V + E) * log(V))."""
    """Space complexity: O(V). We create a pointer array of size V to point to the nodes in the heap, and we create a 
        binary tree of size V to store the nodes. O(V) + O(V) = O(V)."""
    def heap(self, srcIndex):
        # Initialize the binary heap
        heap = BinaryHeap()
        for node in self.network.nodes:
            node.dist = float('inf')
            node.known = False
            node.path = None
            heap.push((node.node_id, node.dist))
        self.network.nodes[srcIndex].dist = 0
        heap.update(srcIndex, 0)

        # Run Dijkstra's algorithm
        while heap.size > 0:
            # Pop the node with the smallest distance
            (node_id, index) = heap.pop()

            node = self.network.nodes[node_id]

            # Update the distances of the neighbors
            for edge in node.neighbors:
                if edge.dest.dist > node.dist + edge.length:
                    edge.dest.dist = node.dist + edge.length
                    edge.dest.path = node
                    heap.update(edge.dest.node_id, edge.dest.dist)
