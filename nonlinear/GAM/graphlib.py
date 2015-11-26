from abc import ABCMeta, abstractmethod

class Graph:
	__metaclass__ = ABCMeta

	@abstractmethod
	def nodes(self):
		return []

	@abstractmethod
	def neighbours(self, node):
		return []

	def rec_sccs(self):
		'''Generates the Strongly Connected Components of the graph.
		   Implemented with the recursive version of Tarjan's algorithm.
		   The rec_sccs function is faster than its iterative version
		   (the sccs function), but it is limited in the number
		   of nodes each component can have, due to the recursion
		   depth limit in python'''

		class TarjanNode:
			def __init__(self, node):
				self.node = node
				self.onstack = False

			def __eq__(self, other):
				return self.node == other.node


		tnodes = {node : TarjanNode(node) for node in self.nodes()}

		try:
			index = self.t_index
		except AttributeError:
			index = None

		self.t_index = 0
		S = []

		def strongconnect(v):
			v.index = self.t_index
			v.low_link = self.t_index
			self.t_index += 1
			S.append(v)
			v.onstack = True

			for w in self.neighbours(v.node):
				w = tnodes[w]
				try:
					if w.index < v.low_link and w.onstack:
						v.low_link = w.index
				except AttributeError:
					for x in strongconnect(w):
						yield x
					if w.low_link < v.low_link:
						v.low_link = w.low_link

			if v.low_link == v.index:
				scc = set()
				w = None
				while w != v:
					w = S[-1]
					del S[-1]
					w.onstack = False
					scc.add(w.node)
				yield scc

		for v in tnodes.values():
			if not hasattr(v, 'index'):
				for scc in strongconnect(v):
					yield scc

		if index == None:
			del self.t_index
		else:
			self.t_index = index


	def sccs(self):
		'''Generates the Strongly Connected Components of the graph.
		   Implemented with the iterative version of Tarjan's algorithm.
		   Each component generated is a set of nodes s.t. for each
		   pair of nodes (u, v) in the component, there exists a path
		   from u to v and another path (possibly the same in reverse
		   order in the case of non-directed graphs) from v to u in
		   the graph.'''

		class TarjanNode:
			def __init__(self, node):
				self.node = node
				self.onstack = False

			def __eq__(self, other):
				return self.node == other.node


		tnodes = {node : TarjanNode(node) for node in self.nodes()}

		try:
			index = self.t_index
		except AttributeError:
			index = None

		self.t_index = 0
		S = []

		def strongconnect(v):
			stack = [v]

			while len(stack) > 0:
				v = stack[-1] # v = stack.top()
				# print 'Accessing stack[' + str(v.node) + ']'
				try:
					w = v.last_child # Throws AttributeError if last_child doesn't exist (only if v has not been visited yet)
					# print 'Last child is', w.node
					if w.low_link < v.low_link:
						v.low_link = w.low_link
					w = tnodes[v.children.next()] # Throws StopIteration if all children of v have been analyzed
					try:
						if w.index < v.low_link and w.onstack: # Throws AttributeError if w.index doesn't exist (only if w has not been visited yet)
							v.low_link = w.index
					except AttributeError:
						# w is a child of v which must be analyzed now
						# same as strongconnect(w)
						v.last_child = w
						# print 'Child of', v.node, 'appended to stack:', w.node
						stack.append(w)

				except StopIteration:
					# Already analyzed all the children of v
					del stack[-1] # stack.pop()
					# print v.node, 'has no more children'
					if v.low_link == v.index:
						scc = set()
						w = None
						while w != v:
							w = S[-1]
							del S[-1]
							w.onstack = False
							scc.add(w.node)
						yield scc

				except AttributeError:
					# First time we analyze v (it does not have last_child attribute)
					v.index = self.t_index
					v.low_link = self.t_index
					self.t_index += 1
					S.append(v)
					v.onstack = True
					# print v.node, 'seems to be a first-timer'
					v.children = iter(self.neighbours(v.node))
					# print 'Its children are shown next:', list(self.neighbours(v.node))
					v.last_child = v # set last_child attribute and let v node be re-analyzed
			# print 'Stack is empty'

		for v in tnodes.values():
			if not hasattr(v, 'index'):
				for scc in strongconnect(v):
					yield scc

		if index == None:
			del self.t_index
		else:
			self.t_index = index




class NiftiGraph(Graph):
	
	def __init__(self, data, pvalue_threshold):
		self.pv = pvalue_threshold
		self.dims = data.shape
		new_dims = map(lambda d: d + 2, self.dims[:3])
		lpv = [self.pv for _ in range(self.dims[2] + 2)]
		matpv = [lpv for _ in range(self.dims[1] + 2)]
		self.data = [matpv] + [[lpv] + [[self.pv] + list(l) + [self.pv] for l in mat] + [lpv] for mat in data] + [matpv]
		
		#	self.data = [matpv]
		#	for mat in data:
		#		mat2 = [lpv]
		#		for l in mat:
		#			mat2.append([self.pv] + list(l) + [self.pv])
		#		mat2.append(lpv)
		#		self.data.append(mat2)
		#	self.data.append(matpv)
		# self.data = self.pv*ones(new_dims)
		# self.data[1 : (new_dims[0] - 1), 1 : (new_dims[1] - 1), 1 : (new_dims[2] - 1)] = data

	def nodes(self):
		for i in range(self.dims[0]):
			for j in range(self.dims[1]):
				for k in range(self.dims[2]):
					yield (i, j, k)

	def neighbours(self, node):
		x, y, z = map(lambda c: c + 1, node)
		if self.data[x][y][z] >= self.pv:
			return
		for dx in (-1, 0, 1):
			for dy in (-1, 0, 1):
				for dz in (-1, 0, 1):
					if dx == 0 and dy == 0 and dz == 0:
						continue
					i, j, k = x + dx, y + dy, z + dz
					if self.data[i][j][k] < self.pv:
						yield (i - 1, j - 1, k - 1)

	def sccs(self):
		''' Generates the Strongly Connected Components of the graph.
			Implemented by using the Breadth First Search algorithm.
			Each component generated is a list of nodes s.t. for each
			pair of nodes (u, v) in the component, there exists a path
			from u to v and another path (possibly the same in reverse
			order in the case of non-directed graphs) from v to u in
			the graph.'''

		visited = [[[False for _ in range(self.dims[2])] for _ in range(self.dims[1])] for _ in range(self.dims[0])]
		for i, j, k in self.nodes():
			if not visited[i][j][k]:
				scc = [(i, j, k)]
				visited[i][j][k] = True
				# Perform a simple bfs and add every node reachable from (i, j, k)
				# to the current strongly connected component (edges are bidirectional)
				index = 0
				while index < len(scc):
					v = scc[index] # v = queue.front()
					index += 1 # queue.pop(); scc.add(v)
					i, j, k = v
					for w in self.neighbours(v):
						x, y, z = w
						if not visited[x][y][z]:
							visited[x][y][z] = True
							scc.append(w) # queue.push(w)
				yield scc



class GenericGraph(Graph):

	def __init__(self, nodes = []):
		self.node_info = []
		self.edges = []
		for node in nodes:
			self.node_info.append(node)
			self.edges.append(set())

	def nodes(self):
		return range(len(self.node_info))

	def neighbours(self, u):
		try:
			return self.edges[u]
		except IndexError:
			return set()

	def add_edge(self, u, v, bidirectional = False):
		if v > len(self.node_info) or v < 0 or u < 0:
			return
		try:
			self.edges[u].add(v)
		except IndexError:
			return
		if bidirectional:
			self.edges[v].add(u)

	def remove_edge(self, u, v, bidirectional = False):
		if v > len(self.node_info) or v < 0 or u < 0:
			return
		try:
			self.edges[u].discard(v)
		except IndexError:
			return
		if bidirectional:
			self.edges[v].discard(u)

	def __repr__(self):
		return 'GenericGraph{ Nodes(' + repr(self.node_info) + ') ; Edges(' + repr(self.edges) + ') }'

	def __str__(self):
		s = 'GenericGraph{\n'
		s += '    Nodes( ' + repr(self.node_info) + ' )\n'
		s += '    Edges(\n'
		for i in range(len(self.edges)):
			stri = str(i)
			s += ' '*(10-len(stri)) + stri + ': ' + repr(self.edges[i]) + '\n'
		s += '    )\n'
		s += '}'
		return s








