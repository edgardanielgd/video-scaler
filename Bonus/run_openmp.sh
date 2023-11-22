for i in 1 2 3 4 5 6 7 8
do
    echo "Running for Threads: $i"
    ./pi_calculus_openmp $i >> $1
    echo "\n\n" >> $1
done