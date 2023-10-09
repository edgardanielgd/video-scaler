import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

FILE_NAME = "logsCuda.txt"

# Cannont be read from file (CUDA is not thread 1 lol)
SEQUENTIAL_TIME = 18119910100

# Indexed by matrix size
data = {}

regex = re.compile( r"Blocks: (\d+) Threads: (\d+) Execution Time: (\d+)" )

def match_lines( reader ):
    return [
        line for line in reader if regex.match( line )
    ]

with open( FILE_NAME, "r" ) as f:

    lines = match_lines( f )

    threads = []
    times = []
    speedups = []

    for line in lines:    
        match = regex.match( line )

        threads.append( int( match.group( 1 ) ) )
        times.append( int( match.group( 2 ) ) )

    threads = np.array( threads )
    times = np.array( times ) / 1000
    speedups = times[ 0 ] / times

    plt.figure()
    
    plt.plot( threads, times, "o-" )

    plt.xlabel( "Number of threads" )
    plt.ylabel( "Time (ms)" )
    plt.title( "Time vs Number of threads" )

    plt.figure()

    plt.plot( threads, speedups, "o-" )

    plt.xlabel( "Number of threads" )
    plt.ylabel( "Speedup" )
    plt.title( "Speedup vs Number of threads" )

    plt.show()

