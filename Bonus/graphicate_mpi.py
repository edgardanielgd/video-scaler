import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

FILE_NAME = "logs_mpi.txt"

# Indexed by matrix size
data = {}


regex = re.compile( r"For (\d+) processes, Pi is approximately .+, Error is .+, Time is (\d+)" )

def match_lines( reader ):
    return [
        line for line in reader if regex.match( line )
    ]

with open( FILE_NAME, "r" ) as f:

    lines = match_lines( f )

    processes_counts = []
    times = []

    for line_id in range(len(lines)):
        line = lines[line_id]
        match = regex.match( line )

        if match is None:
            continue

        processes_counts.append( int(match.group( 1 )) )
        times.append( int( match.group( 2 ) ) )
    
    print( processes_counts )
    print( times )

    processes_counts = np.array( processes_counts )
    times = np.array( times )
    speedups = times[ 0 ] / times

    plt.figure()
    
    plt.plot( processes_counts, times, "o-" )

    plt.xlabel( "Number of processes" )
    plt.ylabel( "Time (us)" )
    plt.title( "Time vs Number of processes" )

    plt.figure()

    plt.plot( processes_counts, speedups, "o-" )

    plt.xlabel( "Number of processes" )
    plt.ylabel( "Speedup" )
    plt.title( "Speedup vs Number of processes" )

    plt.show()

