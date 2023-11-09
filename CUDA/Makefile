OPENCV = `pkg-config --cflags --libs opencv4`

run:
	nvcc mainCuda.cu -o mainCuda -Xcompiler -fopenmp $(OPENCV)
	./mainCuda ./video1.mp4 ./output50.mp4 $(BLOCKS) ${THREADS_PER_BLOCK}