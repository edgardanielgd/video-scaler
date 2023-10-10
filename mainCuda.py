import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

FILE_NAME = "logsCuda.txt"

# Indexed by matrix size
data = {}

regex = re.compile( r"Blocks: (\d+) Threads: (\d+) Execution Time: (\d+)" )

def match_lines( reader ):
    return [
        line for line in reader if regex.match( line )
    ]

with open( FILE_NAME, "r" ) as f:

    lines = match_lines( f )

    blocks = {}
    blocks_speedups = {}

    for line in lines:    
        match = regex.match( line )

        nBlocks = int( match.group( 1 ) )
        nThreads = int( match.group( 2 ) )
        time = int( match.group( 3 ) )

        if nBlocks not in blocks:
            blocks[ nBlocks ] = {}

        blocks[ nBlocks ][ nThreads ] = time
        
    sequential_time = blocks[ 1 ][ 1 ]

    for nBlocks in blocks:
        blocks_speedups[ nBlocks ] = {}
        for nThreads in blocks[ nBlocks ]:
            blocks_speedups[ nBlocks ][ nThreads ] = sequential_time / blocks[ nBlocks ][ nThreads ]

    for nBlocks in blocks:

        threads = []
        times = []
        speedups = []

        for nThreads in blocks[ nBlocks ]:
            threads.append( nThreads )
            times.append( blocks[ nBlocks ][ nThreads ] )
            speedups.append( blocks_speedups[ nBlocks ][ nThreads ] )

        plt.figure()

        plt.plot( threads, times, "o-" )
        plt.xlabel( f"Number of threads per block" )
        plt.ylabel( f"Time (us)" )
        plt.title( f"Time vs number of threads per block. Grid size: {nBlocks}" )

        plt.figure()
        
        plt.plot( threads, speedups, "o-" )
        plt.xlabel( f"Number of threads per block" )
        plt.ylabel( f"Speedup" )
        plt.title( f"Speedup vs number of threads per block. Grid size: {nBlocks}" )
    

    plt.show()

