#!/bin/bash

# Compile code
g++ -fopenmp main.cpp -o main $(pkg-config --cflags --libs opencv4)

# Run code for multiple cases
for i in 1 2 4 8 16
do
    echo "Running for $i threads: ($1, $2, $i)"
    ./main $1 $2 $i >> $3
    echo "\n\n" >> $3
done

# Compile CUDA code
nvcc mainCuda.cu -o main_cuda $(pkg-config --cflags --libs opencv4)

# Run code for multiple cases of different BlockSizes and GridSizes
./main_cuda $1 $2 1 1 >> $4

for i in 10 20 30 40
do
    for j in 4 8 16 32 64
    do
        echo "Running for BlockSize: $i, GridSize: $j"
        ./main_cuda $1 $2 $j $i >> $4
        echo "\n\n" >> $4
    done
done