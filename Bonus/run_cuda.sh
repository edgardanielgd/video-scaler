./pi_calculus_cuda 1 1 >> $1

for i in 512 1024 2048 4096 8192
do
    for j in 8 16 32 64 128 256 512
    do
        echo "Running for Number of threads: $i, GridSize: $j"
        ./pi_calculus_cuda $j $i >> $1
        echo "\n\n" >> $1
    done
done