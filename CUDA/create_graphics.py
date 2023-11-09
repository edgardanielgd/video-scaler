import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

FILE_NAME = "logs.txt"

# Indexed by matrix size
data = {}

regex = re.compile( r"Blocks: (\d+) Threads: (\d+) Processing Time: (\d+) Processing vs duration: (([0-9]*[.])?[0-9]+)" )

def match_lines( reader ):
    return [
        line for line in reader if regex.match( line )
    ]

with open( FILE_NAME, "r" ) as f:

    lines = match_lines( f )

    blocks = {}
    blocks_speedups = {}
    blocks_processing_vs_duration = {}

    for line in lines:    
        match = regex.match( line )

        nBlocks = int( match.group( 1 ) )
        nThreads = int( match.group( 2 ) )
        time = int( match.group( 3 ) )
        processing_vs_duration_ratio = float( match.group( 4 ) )

        if nBlocks not in blocks:
            blocks[ nBlocks ] = {}
            blocks_processing_vs_duration[ nBlocks ] = {}

        blocks[ nBlocks ][ nThreads ] = time
        blocks_processing_vs_duration[ nBlocks ][ nThreads ] = processing_vs_duration_ratio
        
    sequential_time = blocks[ 1 ][ 1 ]

    for nBlocks in blocks:
        blocks_speedups[ nBlocks ] = {}
        for nThreads in blocks[ nBlocks ]:
            blocks_speedups[ nBlocks ][ nThreads ] = sequential_time / blocks[ nBlocks ][ nThreads ]

    for nBlocks in blocks:

        threads = []
        times = []
        speedups = []
        processing_vs_duration_ratios = []

        for nThreads in blocks[ nBlocks ]:
            threads.append( nThreads )
            times.append( blocks[ nBlocks ][ nThreads ] )
            speedups.append( blocks_speedups[ nBlocks ][ nThreads ] )
            processing_vs_duration_ratios.append( blocks_processing_vs_duration[ nBlocks ][ nThreads ] )

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

        plt.figure()
        
        plt.plot( threads, processing_vs_duration_ratios, "o" )
        plt.xlabel( f"Number of threads per block" )
        plt.ylabel( f"Ratio" )
        plt.title( f"Duration vs Processing time ratios. Grid size: {nBlocks}" )
    

    plt.show()

