def main():
	from ttest_ind_mem import *


from memory_profiler import memory_usage
mem_usage = memory_usage((main, [], {}), interval=.1)
print 'Peak memory usage: ' + str(max(mem_usage)) + ' MB'

