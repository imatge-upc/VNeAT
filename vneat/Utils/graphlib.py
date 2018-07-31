from abc import ABCMeta, abstractmethod


class Graph(object):
    __metaclass__ = ABCMeta

    '''[Abstract Class] Defines a Graph by specifying the list of nodes it contains and
        the neighbours that each node has (the outgoing edges). It already implements a
        method to compute the 'Strongly Connected Components' of the graph by using Tarjan's
        algorithm, both in a recursive manner ('rec_sccs') and with an iterative approach
        ('sccs').
    '''

    @abstractmethod
    def nodes(self):
        '''[Abstract Method] Generates/Returns all the nodes of the graph.
        '''
        raise NotImplementedError

    @abstractmethod
    def neighbours(self, node):
        '''
        [Abstract Method] Given a node, generates/returns its neighbours.
        '''
        raise NotImplementedError

    def rec_sccs(self):
        '''Generates the Strongly Connected Components of the graph. Implemented with the
            recursive version of Tarjan's algorithm. The rec_sccs function is faster than
            its iterative version (the sccs function), but it is limited in the number of
            nodes each component can have, due to the recursion depth limit in python.
            Unless you are certain that the graph is small enough or not connected enough
            to have such many connected components and the speed of the method is critical
            for your application, it is recommended that you use the 'sccs' method instead.
        '''

        class TarjanNode:
            def __init__(self, node):
                self.node = node
                self.onstack = False

            def __eq__(self, other):
                return self.node == other.node

        tnodes = {node: TarjanNode(node) for node in self.nodes()}

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
        '''Generates the Strongly Connected Components of the graph. Implemented with the
            iterative version of Tarjan's algorithm. Each component generated is a set of
            nodes s.t. for each pair of nodes (u, v) in the component, there exists a path
            from u to v and another path (possibly the same in reverse order in the case of
            non-directed graphs) from v to u in the graph.
        '''

        class TarjanNode:
            def __init__(self, node):
                self.node = node
                self.onstack = False

            def __eq__(self, other):
                return self.node == other.node

        tnodes = {node: TarjanNode(node) for node in self.nodes()}

        try:
            index = self.t_index
        except AttributeError:
            index = None

        self.t_index = 0
        S = []

        def strongconnect(v):
            stack = [v]

            while len(stack) > 0:
                v = stack[-1]  # v = stack.top()
                # print 'Accessing stack[' + str(v.node) + ']'
                try:
                    w = v.last_child  # Throws AttributeError if last_child doesn't exist (only if v has not been visited yet)
                    # print 'Last child is', w.node
                    if w.low_link < v.low_link:
                        v.low_link = w.low_link
                    w = tnodes[v.children.next()]  # Throws StopIteration if all children of v have been analyzed
                    try:
                        if w.index < v.low_link and w.onstack:  # Throws AttributeError if w.index doesn't exist (only if w has not been visited yet)
                            v.low_link = w.index
                    except AttributeError:
                        # w is a child of v which must be analyzed now
                        # same as strongconnect(w)
                        v.last_child = w
                        # print 'Child of', v.node, 'appended to stack:', w.node
                        stack.append(w)

                except StopIteration:
                    # Already analyzed all the children of v
                    del stack[-1]  # stack.pop()
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
                    v.last_child = v  # set last_child attribute and let v node be re-analyzed
                # print 'Stack is empty'

        for v in tnodes.values():
            if not hasattr(v, 'index'):
                for scc in strongconnect(v):
                    yield scc

        if index == None:
            del self.t_index
        else:
            self.t_index = index


class UndirectedClustering3DGraph(Graph):
    def __init__(self, data, not_fulfilling_element):
        super(UndirectedClustering3DGraph, self).__init__()
        self.dims = data.shape
        new_dims = map(lambda d: d + 2, self.dims[:3])
        self.default = not_fulfilling_element
        l = [self.default for _ in range(self.dims[2] + 2)]
        mat = [l for _ in range(self.dims[1] + 2)]
        self.data = [mat] + [[l] + [[self.default] + list(l) + [self.default] for l in mat] + [l] for mat in data] + [
            mat]

    @abstractmethod
    def __condition_function__(self, elem):
        raise NotImplementedError

    def nodes(self):
        return ((i, j, k) for i in range(self.dims[0]) for j in range(self.dims[1]) for k in range(self.dims[2]))

    def neighbours(self, node):
        x, y, z = map(lambda c: c + 1, node)
        if not self.__condition_function__(self.data[x][y][z]):
            return
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    i, j, k = x + dx, y + dy, z + dz
                    if self.__condition_function__(self.data[i][j][k]):
                        yield (i - 1, j - 1, k - 1)

    def sccs(self):
        '''Generates the Strongly Connected Components of the graph. Implemented by using the
            Breadth First Search algorithm. Each component generated is a list of nodes s.t.
            for each pair of nodes (u, v) in the component, there exists a path from u to v
            in the graph (these nodes are strongly connected since this is a non-directed
            graph and thus the same path in reverse order also exists).
        '''

        visited = [[[False for _ in range(self.dims[2])] for _ in range(self.dims[1])] for _ in range(self.dims[0])]
        for i, j, k in self.nodes():
            if not visited[i][j][k]:
                scc = [(i, j, k)]
                visited[i][j][k] = True
                # Perform a simple bfs and add every node reachable from (i, j, k)
                # to the current strongly connected component (edges are bidirectional)
                index = 0
                while index < len(scc):
                    v = scc[index]  # v = queue.front()
                    index += 1  # queue.pop(); scc.add(v)
                    i, j, k = v
                    for w in self.neighbours(v):
                        x, y, z = w
                        if x >= self.dims[0] or y >= self.dims[1] or z >= self.dims[2] or x<0 or y<0 or z<0:
                            break

                        if not visited[x][y][z]:
                            visited[x][y][z] = True
                            scc.append(w)  # queue.push(w)
                yield scc


class NiftiGraph(UndirectedClustering3DGraph):
    def __init__(self, data, lower_threshold=None, upper_threshold=None):
        self.lt = lower_threshold
        self.ut = upper_threshold

        if not (self.lt is None):
            not_fulfilling_element = self.lt - 1
        elif not (self.ut is None):
            not_fulfilling_element = self.ut + 1
        else:
            not_fulfilling_element = 0.0

        super(NiftiGraph, self).__init__(data, not_fulfilling_element)

    def __condition_function__(self, elem):
        if (not (self.lt is None)) and (elem < self.lt):
            return False
        if (not (self.ut is None)) and (elem >= self.ut):
            return False
        return True

    def sccs(self):
        '''Generates the Strongly Connected Components of the graph. Implemented by using the
            Breadth First Search algorithm. Each component generated is a list of nodes s.t.
            for each pair of nodes (u, v) in the component, there exists a path from u to v
            in the graph (these nodes are strongly connected since this is a non-directed
            graph and thus the same path in reverse order also exists).
        '''

        if self.lt is None and self.ut is None:
            yield list(self.nodes())
            return

        for scc in super(NiftiGraph, self).sccs():
            yield scc


class GenericGraph(Graph):
    def __init__(self, nodes=[], directed=True):
        Graph.__init__(self)
        if isinstance(nodes, int):
            nodes = [None] * nodes
        self._nodes = []
        self._edges = []
        for node in nodes:
            self._nodes.append(node)
            self._edges.append({})

        self._directed = directed

    @property
    def directed(self):
        return self._directed

    def nodes(self):
        i = 0
        while i < len(self._nodes):
            yield i
            i += 1

    def node(self, u):
        return self._nodes[u]

    def edges(self):
        return ((u, v) for u in self.nodes() for v in self._edges[u])

    def edge(self, u, v):
        if u < 0:
            # Otherwise this goes undetected since python
            # lists accept negative indices
            raise KeyError
        return self._edges[u][v]

    def neighbours(self, u):
        if u < 0:
            return set()
        try:
            return self._edges[u].iterkeys()
        except IndexError:
            return set()

    def add_edge(self, u, v, info=None):
        if u < 0 or v < 0:
            # Otherwise this goes undetected since python
            # lists accept negative indices
            raise KeyError
        try:
            _ = self._nodes[v]  # raise IndexError if v is invalid
            self._edges[u][v] = info
            if not self._directed:
                self._edges[v][u] = info
        except IndexError:
            raise KeyError

    def remove_edge(self, u, v):
        if v < 0 or u < 0:
            # Otherwise this goes undetected since python
            # lists accept negative indices
            raise KeyError
        try:
            del self._edges[u][v]
            if not self._directed:
                del self._edges[v][u]
        except IndexError:
            raise KeyError

    def sccs(self):
        if self._directed:
            # use Tarjan's algorithm implemented in a superclass (Graph)
            for scc in super(GenericGraph, self).sccs():
                yield scc
        else:
            # perform bfs for each node
            visited = [False] * len(self._nodes)
            for u in self.nodes():
                if not visited[u]:
                    scc = [u]
                    visited[u] = True
                    # Perform a simple bfs and add every node reachable from u to the current strongly
                    # connected component (edges are undirected)
                    index = 0
                    while index < len(scc):
                        v = scc[index]  # v = queue.front()
                        index += 1  # queue.pop(); scc.add(v)
                        for w in self.neighbours(v):
                            if not visited[w]:
                                visited[w] = True
                                scc.append(w)  # queue.push(w)
                    yield scc

    def __repr__(self):
        return 'GenericGraph{ Nodes(' + repr(self._nodes) + ') ; Edges(' + repr(self._edges) + ') }'

    def __str__(self):
        s = 'GenericGraph{\n'
        s += '    Nodes( \n'
        for u in self.nodes():
            stru = str(u)
            s += ' ' * (14 - len(stru)) + stru + ':   ' + repr(self.node(u)) + '\n'
        s += ')\n'
        s += '    Edges(\n'
        for u, v in self.edges():
            stru = str(u)
            strv = str(v)
            s += ' ' * (7 - len(stru)) + stru + ' -> '
            s += strv + ' ' * (7 - len(strv)) + ':    '
            s += repr(self.edge(u, v)) + '\n'
        s += '    )\n'
        s += '}'
        return s
