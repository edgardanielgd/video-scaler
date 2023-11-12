OPENCV = `pkg-config --cflags --libs opencv4`

# Execute from WSL
install-mpi:
	apt-get update
	adduser mpiuser --uid 99
	apt-get install openmpi-bin openmpi-common libopenmpi-dev
	apt-get install nfs-kernel-server
	apt-get install nfs-common
	echo "/home/mpiuse *(rw,sync,no_subtree_check)" > /etc/exports
	service nfs-kernel-server restart
	exportfs -a