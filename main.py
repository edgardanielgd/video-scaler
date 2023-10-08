import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

FILE_NAME = "omp_multi_matricial_test.txt"

# Indexed by matrix size
data = {}

with open( FILE_NAME, "r" ) as f:

    

    for i, matrix_size in enumerate( data ):

        plt.figure()
        
        plt.plot(
            data[ matrix_size ][ "thread_numbers" ],
            data[ matrix_size ][ "threads_times" ],
            "o-"
        )

        plt.xlabel( "Number of threads" )
        plt.ylabel( "Time (ns)" )
        plt.title( "Time vs Number of threads for matrix size {}".format( matrix_size ) )

        plt.figure()

        plt.plot(
            data[ matrix_size ][ "thread_numbers" ],
            data[ matrix_size ][ "threads_speedups" ],
            "o-"
        )

        plt.xlabel( "Number of threads" )
        plt.ylabel( "Speedup" )
        plt.title( "Speedup vs Number of threads for matrix size {}".format( matrix_size ) )

    plt.show()

