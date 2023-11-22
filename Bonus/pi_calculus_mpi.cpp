#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <chrono>
#include <iostream>

#define ITERATIONS 2e09

int calculatePi(double *pi, int ID, int numprocs)
{
    int start, end;
    start = (ITERATIONS / numprocs) * ID;
    end = (ITERATIONS / numprocs) + 1;
    int i = start;

    do
    {
        *pi = *pi + (double)(4.0 / ((i * 2) + 1));
        i++;
        *pi = *pi - (double)(4.0 / ((i * 2) + 1));
        i++;
    } while (i < end);

    return 0;
}

using namespace std;

int main(int argc, char *argv[])
{
    auto start = chrono::high_resolution_clock::now();

    int done = 0, n, processId, numprocs, I, rc, i;
    double PI25DT = 3.141592653589793238462643;
    double local_pi, global_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    global_pi = 0.0;
    calculatePi(&local_pi, processId, numprocs);

    printf("%i ", processId);
    fflush(stdout);

    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (processId == 0)
    {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);

        cout << "\nFor " << numprocs
             << " processes, Pi is approximately " << global_pi
             << ", Error is " << fabs(global_pi - PI25DT)
             << ", Time is " << duration.count() << endl;
    }

    MPI_Finalize();
    return 0;
}