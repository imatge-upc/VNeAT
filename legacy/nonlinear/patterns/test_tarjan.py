from graphlib import GenericGraph

print 'SCCs in first graph'
g1 = GenericGraph(5)
g1.add_edge(1, 0)
g1.add_edge(0, 2)
g1.add_edge(2, 1)
g1.add_edge(0, 3)
g1.add_edge(3, 4)
for x in g1.sccs():
    print list(x)

print

print 'SCCs in second graph'
g2 = GenericGraph(4)
g2.add_edge(0, 1)
g2.add_edge(1, 2)
g2.add_edge(2, 3)
for x in g2.sccs():
    print list(x)

print

print 'SCCs in third graph'
g3 = GenericGraph(7)
g3.add_edge(0, 1)
g3.add_edge(1, 2)
g3.add_edge(2, 0)
g3.add_edge(1, 3)
g3.add_edge(1, 4)
g3.add_edge(1, 6)
g3.add_edge(3, 5)
g3.add_edge(4, 5)
for x in g3.sccs():
    print list(x)

print

print 'SCCs in fourth graph'
g4 = GenericGraph(11)
g4.add_edge(0, 1)
g4.add_edge(0, 3)
g4.add_edge(1, 2)
g4.add_edge(1, 4)
g4.add_edge(2, 0)
g4.add_edge(2, 6)
g4.add_edge(3, 2)
g4.add_edge(4, 5)
g4.add_edge(4, 6)
g4.add_edge(5, 6)
g4.add_edge(5, 7)
g4.add_edge(5, 8)
g4.add_edge(5, 9)
g4.add_edge(6, 4)
g4.add_edge(7, 9)
g4.add_edge(8, 9)
g4.add_edge(9, 8)
for x in g4.sccs():
    print list(x)

print

print 'SCCs in fifth graph'
g5 = GenericGraph(5)
g5.add_edge(0, 1)
g5.add_edge(1, 2)
g5.add_edge(2, 3)
g5.add_edge(2, 4)
g5.add_edge(3, 0)
g5.add_edge(4, 2)
for x in g5.sccs():
    print list(x)

print

#	Expected output:
#	
#	SCCs in first graph
#	4
#	3
#	0 1 2
#
#	SCCs in second graph
#	3
#	2
#	1
#	0
#
#	SCCs in third graph
#	5
#	3
#	4
#	6
#	0 1 2
#
#	SCCs in fourth graph
#	8 9
#	7
#	4 5 6
#	0 1 2 3
#	10
#
#	SCCs in fifth graph
#	0 1 2 3 4
