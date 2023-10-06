#!/bin/bash

# Compile code
g++ -fopenmp main.cpp -o main $(pkg-config --cflags --libs opencv4)

# Run code for multiple cases
for i in 1, 2, 4, 8, 16
do
    echo "Running for $i threads: ($1, $2, $i)"
done