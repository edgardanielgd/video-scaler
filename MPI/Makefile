OPENCV = -I/usr/local/include/opencv4 -L/usr/local/bin -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio

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

install-opencv-3:
	sudo apt-get install build-essential
	sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	git clone https://github.com/opencv/opencv.git
	git clone https://github.com/opencv/opencv_contrib.git
	cd opencv
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_DOCS=ON \
      -D BUILD_EXAMPLES=ON \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D OPENCV_GENERATE_PKGCONFIG=YES \
      -D OPENCV_EXTRA_EXE_LINKER_FLAGS=-latomic \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON_EXECUTABLE=$(which python2) \
      ..
	make -j4
	sudo make install
	export PKG_CONFIG_PATH=$(pwd)/unix-install/

install-opencv-4:
	sudo apt update && sudo apt install -y cmake g++ wget unzip
	wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
	unzip opencv.zip
	mkdir -p build && cd build
	cmake  ../opencv-4.x
	cmake --build .
	sudo make install
	touch /etc/ld.so.conf.d/opencv.conf
	vim /etc/ld.so.conf.d/opencv.conf
	sudo ldconfig -v

compile-mpi:
	mpic++ -o main main.cpp $(OPENCV)

run-mpi:
	mpirun -np 2 --allow-run-as-root ./main 