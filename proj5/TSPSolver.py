#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import queue
from Node import Node

class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			route = []
			# sort the edges by cost
			edges = []
			for i in range(ncities):
				for j in range(i + 1, ncities):
					cost = cities[i].costTo(cities[j])
					if cost != np.inf:
						edges.append((cost, i, j))
			edges.sort()
			route = [edges[0][1], edges[0][2]]
			route = self.greedy_helper(route, edges, ncities, cities)
			tries = 1
			while route is None and tries < len(edges):
				# if the greedy helper returns None, then we need to try a new starting point
				route = [edges[tries][1], edges[tries][2]]
				route = self.greedy_helper(route, edges, ncities, cities)
				tries += 1
			final_route = []
			for i in range(ncities):
				final_route.append(cities[route[i]])
			bssf = TSPSolution(final_route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def greedy_helper(self, route, edges, ncities, cities):
		# implement a recursive greedy algorithm that go through each edge and adds the next closest city
		restricted_path = []
		for edge in edges:
			if edge[1] == route[-1] and edge[2] not in restricted_path and edge[2] not in route:
				route.append(edge[2])
				if len(route) == ncities:
					# special final check to make sure the route is complete
					if cities[route[-1]].costTo(cities[route[0]]) != np.inf:
						return route
				else:
					add_route = self.greedy_helper(route, edges, ncities, cities)
					if add_route is None:
						restricted_path.append(route.pop())
						# if len(route) == MAX_GREEDY_DEPTH:
						# 	return False
						continue
					else:
						return add_route
			elif edge[2] == route[-1] and edge[1] not in restricted_path and edge[1] not in route:
				route.append(edge[1])
				if len(route) == ncities:
					# special final check to make sure the route is complete
					if cities[route[-1]].costTo(cities[route[0]]) != np.inf:
						return route
				else:
					add_route = self.greedy_helper(route, edges, ncities, cities)
					if add_route is None:
						restricted_path.append(route.pop())
						# if len(route) == MAX_GREEDY_DEPTH:
						# 	return False
						continue
					else:
						return add_route
		return None

	def broken_greedy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		tries = 0
		while not foundTour and time.time() - start_time < time_allowance:
			route = []
			# sort the edges by cost
			edges = []
			for i in range(ncities):
				for j in range(i + 1, ncities):
					cost = cities[i].costTo(cities[j])
					if cost != np.inf:
						edges.append((cost, i, j))
			edges.sort(key=lambda x: x[0])
			# create a list of all the cities
			remaining = list(range(ncities))
			# start at the first city
			current = remaining.pop(0 + tries)  # added count to make it more random
			route.append(cities[current])
			# while there are still cities to visit
			while len(remaining) > 0:
				# find the next closest city
				found_edge = False
				for edge in edges:
					if edge[1] == current:
						if edge[2] in remaining:
							current = edge[2]
							found_edge = True
							break
				if not found_edge:
					route = []
					break
				# remove it from the list of remaining cities
				remaining.remove(current)
				# add it to the route
				route.append(cities[current])
			if not route:
				tries += 1
				continue
			bssf = TSPSolution(route)
			if bssf.cost < np.inf:
				count += 1
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	# Time complexity: O(S * L), where S is the total number of states created and L is the complexity of computing
	# the lower bound (which is O(n^2), iterating through the whole matrix). The worst case for S would be exploring
	# the whole tree, which would be O(b^n) where b is the average branching factor and n is the depth. However,
	# pruning allows it to not explore the whole tree, so depending on how well I've implemented it, it could be much
	# less than that.

	# Space complexity: O(T * max_queue_size), where T is the size of the data structure for each state (which is O(
	# n^2), for the matrix).
	def branchAndBound(self, time_allowance=60.0):
		# initialize tracking variables
		results = {}
		count = 0
		max_queue_size = 1
		total_states = 1
		total_pruned = 0
		init_results = self.defaultRandomTour(time_allowance)
		bssf = init_results['soln']
		cities = self._scenario.getCities()
		ncities = len(cities)
		start_time = time.time()

		# create a matrix of the distances between cities
		matrix = np.zeros((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				matrix[i][j] = cities[i].costTo(cities[j])

		# initial reduction of the matrix
		# for each row, subtract the smallest value from each element
		cost = 0
		for i in range(ncities):
			row_min = np.min(matrix[i])
			cost += row_min
			matrix[i] = matrix[i] - row_min
		# for each column, subtract the smallest value from each element
		for i in range(ncities):
			col_min = np.min(matrix[:, i])
			cost += col_min
			matrix[:, i] = matrix[:, i] - col_min

		# create a priority queue
		pq = []
		remaining = list(range(ncities))
		current = remaining.pop(0)
		root = Node(matrix, cost, [current], remaining)
		heapq.heappush(pq, (root.cost, root))

		# begin the search
		while len(pq) > 0 and time.time() - start_time < time_allowance:
			# get the next node to expand
			cost, node = heapq.heappop(pq)
			# if the node is a leaf node
			if len(node.remaining) == 0:
				count += 1
				# if the node is a better solution than the current best solution
				if node.cost < bssf.cost:
					# update the best solution
					route = []
					for city in node.path:
						route.append(cities[city])
					bssf = TSPSolution(route)
					# prune the queue
					for item in pq:
						if item[1].cost >= bssf.cost:
							pq.remove(item)
							total_pruned += 1
					# if the time limit has been reached
					if time.time() - start_time > time_allowance:
						# return the best solution found so far
						break
			# if the node is not a leaf node
			else:
				# if the node is a better solution than the current best solution
				children, num_pruned, num_created = self.expand_node(node, bssf)
				total_states += num_created
				total_pruned += num_pruned
				for child in children:
					# add the child to the priority queue
					heapq.heappush(pq, (child.cost * len(child.remaining), child))
				if len(pq) > max_queue_size:
					max_queue_size = len(pq)

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_queue_size
		results['total'] = total_states
		results['pruned'] = total_pruned
		return results

	# Time complexity: O(n^2), where n is the number of cities. We have to iterate through the whole matrix to reduce it.

	# Space complexity: O(n^2), where n is the number of cities. We are only storing the reduced 2D matrix.
	def reduce_matrix(self, node):
		# reduce the matrix by subtracting the minimum value in each row from each value in the row
		# and subtracting the minimum value in each column from each value in the column
		# return the reduced matrix and the total cost of the reductions
		matrix = node.matrix
		path = node.path

		cost = 0
		# subtract the minimum value in each row from each value in the row
		for i in range(len(matrix)):
			row_min = np.min(matrix[i])
			if row_min == np.inf:
				if i not in path:
					return None, np.inf
				else:
					continue
			matrix[i] -= row_min
			cost += row_min

		# subtract the minimum value in each column from each value in the column
		for i in range(len(matrix)):
			col_min = np.min(matrix[:, i])
			if col_min == np.inf:
				if i not in path:
					return None, np.inf
				else:
					continue
			matrix[:, i] -= col_min
			cost += col_min
		# return the reduced matrix and the total cost of the reductions
		return matrix, cost

	# Time complexity: O(n^3). The first for loop iterates through the remaining list, and then we must reduce the
	# matrix for each child node created. The length of the remaining list will worst case be length n, and reducing
	# takes O(n^2) time, giving us a total of O(n^3) time.

	# Space complexity: O(n^3). The 2D matrix is created for each child node (n^2) and stored in the children list
	# (length n).
	def expand_node(self, node, bssf):
		# create a list of all the children of the node
		children = []
		num_pruned = 0
		num_created = 0
		# for each city that has not been visited
		for city in node.remaining:
			# create a child node
			num_created += 1
			child_path = node.path.copy()
			former_city = child_path[-1]
			child_path.append(city)
			child_remaining = node.remaining.copy()
			child_remaining.remove(city)
			child = Node(np.copy(node.matrix), 0, child_path, child_remaining)

			# get initial cost of the parent node + traveling to the city
			cost = node.cost
			cost += node.matrix[former_city][city]

			# set the row and column of the child node's matrix to infinity
			child.matrix[former_city] = np.inf
			child.matrix[:, city] = np.inf

			# reduce the child node's matrix and add the cost of the reductions to the child node's cost
			child.matrix, child.cost = self.reduce_matrix(child)
			child.cost += cost

			if child.cost > bssf.cost:
				num_pruned += 1
				continue
			# add the child node to the list of children
			children.append(child)
		# return the list of children
		return children, num_pruned, num_created

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy(self, time_allowance=60.0):
		pass
