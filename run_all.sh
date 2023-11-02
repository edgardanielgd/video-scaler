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

for i in 40 50 60 70
do
    for j in 128 256 512 1024
    do
        echo "Running for BlockSize: $j, GridSize: $i"
        ./main_cuda $1 $2 $j $i >> $4
        echo "\n\n" >> $4
    done
done