OPENCV = -I/usr/local/include/opencv4 -L/usr/local/bin -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio

# Execute from WSL
install-mpi:
	apt-get update
	sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

install-opencv-4:
	sudo apt-get install build-essential
	sudo apt-get install pkg-config cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev ffmpeg
	wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
	unzip opencv.zip
	cd opencv-4.x
	mkdir -p build && cd build
	cmake  ../opencv-4.x
	cmake --build .
	sudo make install
	sudo touch /etc/ld.so.conf.d/opencv.conf
	sudo vim /etc/ld.so.conf.d/opencv.conf
	sudo ldconfig -v

fix-ldconfig:
	sudo touch /etc/ld.so.conf.d/opencv.conf
	sudo vim /etc/ld.so.conf.d/opencv.conf
	# Add /usr/local/lib to the file
	sudo ldconfig -v

configure-ssh:
	cat ssh.config >> ~/.ssh/config

compile-mpi:
	mpic++ -o /main main.cpp $(OPENCV)

run-mpi:
	mpirun -np 2 --allow-run-as-root --hostfile ./mpi_hosts main 

add-ssh-key:
	ssh-agent bash
	ssh-add SSHHosts.pem