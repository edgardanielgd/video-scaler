#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <chrono>
#include <iostream>

#define MAX_TERMS 2000000000

using namespace std;

int main(int argc, char **argv)
{
    auto start = chrono::high_resolution_clock::now();

    if (argc != 2)
    {
        printf("usage: ./pi thread_count");
        return 0;
    }

    int thread_count = atoi(argv[1]);

    double result = 0;

#pragma omp parallel for num_threads(thread_count) reduction(+ : result)
    for (int i = 0; i < MAX_TERMS; ++i)
    {
        result += 4.0 * (i % 2 == 0 ? 1 : -1) / (2.0 * i + 1);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);

    cout << "Result " << result
         << ", Thread count " << thread_count
         << ", Time is " << duration.count()
         << " nanoseconds"
         << endl;

    return 0;
}