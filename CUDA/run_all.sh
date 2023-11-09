#!/bin/bash

# Compile CUDA code
nvcc mainCuda.cu -o main_cuda $(pkg-config --cflags --libs opencv4)

# Run code for multiple cases of different BlockSizes and GridSizes
./main_cuda $1 $2 1 1 >> $3

for i in 10 20 40 80 160
do
    for j in 4 16 32 64 128 256 512
    do
        echo "Running for BlockSize: $j, GridSize: $i"
        ./main_cuda $1 $2 $j $i >> $3
        echo "\n\n" >> $3
    done
done