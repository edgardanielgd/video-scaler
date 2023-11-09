OPENCV = `pkg-config --cflags --libs opencv4`

run:
	g++ -fopenmp main.cpp -o main $(OPENCV)
	./main ./video1.mp4 ./output50.mp4 $(THREADS)
