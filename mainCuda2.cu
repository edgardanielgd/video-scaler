%%writefile mainCuda.cu

#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <chrono>
#include "omp.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BLOCKS 40
#define NUM_THREADS_PER_BLOCK 64
#define DEFAULT_FILENAME "video1.mp4"
#define DEFAULT_OUTPUT_FILENAME "output.mp4"
#define DEFAULT_OUTPUT_WIDTH 640
#define DEFAULT_OUTPUT_HEIGHT 360

#define NEAREST_NEIGHBOR 0
#define BILINEAR_INTERPOLATION 1
#define BICUBIC_INTERPOLATION 2

#define CUDA_FRAME_BATCH_SIZE

using namespace std;

char *filename;
char *output_filename;
int input_width, output_width;
int input_height, output_height;
int num_threads_per_block, num_blocks;
int method;

__global__ void
process(const unsigned char *input, unsigned char *output, int frameCount, int inp_w, int inp_h, int out_w, int out_h )
{
    // Process one frame per thread
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int pixels_count = out_w * out_h;

    // Process one pixel at a time in parallel when possible, otherwise process modular pixels
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            int x = (int)(i * (inp_h / out_h));
            int y = (int)(i * (inp_w / out_w));

            // Average 2x2 neighbors values
            int r = 0, g = 0, b = 0;
            int count = 0;

            for (int xn = x; xn <= min(x, inp_h - 1); xn++)
            {
                for (int yn = y; yn <= min(y, inp_w - 1); yn++)
                {
                    r += input[(xn * inp_w + yn) * 3];
                    g += input[(xn * inp_w + yn) * 3 + 1];
                    b += input[(xn * inp_w + yn) * 3 + 2];

                    count++;
                }
            }

            int dest_x = (int)(src_i * (inp_h / out_h));
            int dest_y = (int)(src_j * (inp_w / out_w));

            output[(dest_x * out_w + dest_y) * 3] = r / count;
            output[(dest_x * out_w + dest_y) * 3 + 1] = g / count;
            output[(dest_x * out_w + dest_y) * 3 + 2] = b / count;
        }
    }
}

void process_video( cv::Mat frames[],int frameCount){

    cudaError_t err = cudaSuccess;

    // Process one frame at a time in parallel
    cv::Mat output[frameCount];

    // Allocate Unified Memory – accessible from CPU or GPU
    int numElementsOutput = output_width * output_height * 3;
    int numElementsInput = input_width * input_height * 3;

    size_t sizeInput = numElementsInput * sizeof(unsigned char);
    size_t sizeOutput = numElementsOutput * sizeof(unsigned char);

    unsigned char *h_Output = (unsigned char *)malloc(sizeOutput * frameCount);
    unsigned char *h_Input = (unsigned char *)malloc(sizeInput * frameCount);

    if( h_Input == NULL || h_Output == NULL ){
        printf("Error allocating memory [HOST]\n");
        exit(1);
    }

    // Process frame in parallel
    unsigned char *d_Input, *d_Output;

    // Prepare input data on device
    err = cudaMalloc((void **)&d_Input, sizeInput * frameCount);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Input, h_Input, sizeInput * frameCount, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the output data on the device
    err = cudaMalloc((void **)&d_Output, sizeOutput * frameCount);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Process frame 
    process<<<num_blocks, num_threads_per_block>>>(
        d_Input, d_Output, frameCount,
        input_width, input_height,
        output_width, output_height
    );

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    err = cudaMemcpy(h_Output, d_Output, sizeOutput * frameCount, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_Input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_Output);
    free(h_Input);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    // Get params size
    if (argc >= 5)
    {
        filename = argv[1];
        output_filename = argv[2];
        num_threads_per_block = atoi(argv[3]);
        num_blocks=atoi(argv[4]);
    }
    else
    {
        filename = (char *)DEFAULT_FILENAME;
        output_filename = (char *)DEFAULT_OUTPUT_FILENAME;
        num_threads_per_block = NUM_THREADS_PER_BLOCK;
        num_blocks=NUM_BLOCKS;
    }
    if (argc >= 7)
    {
        output_width = atoi(argv[4]);
        output_height = atoi(argv[5]);
    }
    else
    {
        output_width = DEFAULT_OUTPUT_WIDTH;
        output_height = DEFAULT_OUTPUT_HEIGHT;
    }

    cv::VideoCapture capture(filename);

    if (!capture.isOpened())
    {
        return 1;
    }

    // Saving some video metadata
    input_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    input_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int fps = capture.get(cv::CAP_PROP_FPS);

    cout << "Video metadata: (" << num_threads_per_block << ")" << endl;
    cout << "  - Frame width: " << input_width << endl;
    cout << "  - Frame height: " << input_height << endl;
    cout << "  - Frame count: " << frame_count << endl;
    cout << "  - FPS: " << fps << endl;

    cout << "Output video metadata:" << endl;
    cout << "  - Frame width: " << output_width << endl;
    cout << "  - Frame height: " << output_height << endl;

    cout << "Input file: " << filename << endl;
    cout << "Output file: " << output_filename << endl;

    assert(input_width >= output_width && input_height >= output_height);

    // Copying video frame by frame into video writter
    cv::VideoWriter writter(output_filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(output_width, output_height));

    // We'll write output video frame by frame in parallel
    cv::Mat frames[frame_count];

    cout << "Reading video frames..." << endl;
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        capture >> frames[frame_number];
    }

    capture.release();

    auto start = chrono::high_resolution_clock::now();

    cout << "Processing video..." << endl;

    process_video(frames, frame_count);

    cout << "Done" << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Blocks: " << num_blocks << "Threads: " << num_threads_per_block << " Execution Time: " << duration.count() << endl;

    // Joining all frames into a single video

    cout << "Writing video..." << endl;
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        writter << frames[frame_number];
    }
    cout << "Done" << endl;

    writter.release();

    printf("Done\n");
    return 0;
}
