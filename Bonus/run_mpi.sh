# echo "Running for 1 processes" >> $1
# mpirun -np 1 -H 172.31.22.155 pi_calculus_mpi >> $1
# echo "Running for 2 processes" >> $1
# mpirun -np 2 -H 172.31.22.155,172.31.22.65 pi_calculus_mpi >> $1
# echo "Running for 3 processes" >> $1
# mpirun -np 3 -H 172.31.22.155,172.31.22.65,172.31.22.74 pi_calculus_mpi >> $1
# echo "Running for 4 processes" >> $1
mpirun -np 4 -H 172.31.22.155,172.31.22.65,172.31.22.74,172.31.29.7 pi_calculus_mpi >> $1
echo "Running for 5 processes" >> $1
mpirun -np 5 -H 172.31.22.155,172.31.22.65,172.31.22.74,172.31.29.7,172.31.16.234 pi_calculus_mpi >> $1
echo "Running for 6 processes" >> $1
mpirun -np 6 -H 172.31.22.155,172.31.22.65,172.31.22.74,172.31.29.7,172.31.16.234,172.31.18.8 pi_calculus_mpi >> $1
echo "Running for 7 processes" >> $1
mpirun -np 7 -H 172.31.22.155,172.31.22.65,172.31.22.74,172.31.29.7,172.31.16.234,172.31.18.8,172.31.22.78 pi_calculus_mpi >> $1
echo "Running for 8 processes" >> $1
mpirun -np 8 -H 172.31.22.155,172.31.22.65,172.31.22.74,172.31.29.7,172.31.16.234,172.31.18.8,172.31.22.78,172.31.24.22 pi_calculus_mpi >> $1