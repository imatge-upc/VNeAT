from sys import argv

if len(argv) != 2:
    print 'Usage: python ' + argv[0] + ' file'
else:
    def main(filename, ):
        try:
            execfile(filename)
        except Exception as e:
            print 'There was an error while executing the file ' + filename
            print 'Details:'
            print e
            # Lose some time to avoid memory_usage to call this function again
            x = 0
            while x < 10 ** 7:
                x += 1


    from memory_profiler import memory_usage

    mem_usage = memory_usage((main, [argv[1]], {}), interval=.1)
    print 'Peak memory usage: ' + str(max(mem_usage)) + ' MB'
