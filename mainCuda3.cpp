% % writefile mainCuda.cu

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

// Number of frames to be passed to GPU on step
#define FRAMES_BATCH_SIZE 100

    char *filename;
char *output_filename;
int input_width, output_width;
int input_height, output_height;
int num_threads_per_block, num_blocks;
int method;

__global__ void
process(
    const unsigned char *input_data, unsigned char *output_data, int pixels_per_thread,
    int frame_count, int inp_w, int inp_h, int out_w, int out_h)
{
    // Process one frame per thread
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int pixel_index = index * pixels_per_thread; pixel_index < (index + 1) * pixels_per_thread; pixel_index++)
    {
        if (pixel_index >= out_w * out_h * frame_count)
        {
            return;
        }

        // Calculate pixel coordinates
        int output_frame_index = pixel_index / (out_w * out_h);
        int output_pixel_index_in_frame = pixel_index % (out_w * out_h);

        int output_pixel_x = output_pixel_index_in_frame / out_w;
        int output_pixel_y = output_pixel_index_in_frame % out_w;

        // Perform actual transformation
        int estimated_input_pixel_x = (int)(output_pixel_x * (inp_h / out_h));
        int estimated_input_pixel_y = (int)(output_pixel_y * (inp_w / out_w));

        // Average 2x2 neighbors values
        int r = 0, g = 0, b = 0;
        int count = 0;

        int xn_limit = estimated_input_pixel_x < inp_h - 1 ? estimated_input_pixel_x : inp_h - 1;
        int yn_limit = estimated_input_pixel_y < inp_w - 1 ? estimated_input_pixel_y : inp_w - 1;

        for (int xn = estimated_input_pixel_x; xn <= xn_limit; xn++)
        {
            for (int yn = estimated_input_pixel_y; yn <= yn_limit; yn++)
            {
                int input_data_offset = output_frame_index * inp_w * inp_h * 3 + xn * inp_w * 3 + yn * 3;
                r += input_data[input_data_offset];
                g += input_data[input_data_offset + 1];
                b += input_data[input_data_offset + 2];

                count++;
            }
        }

        r /= count;
        g /= count;
        b /= count;

        // Write result to output array
        int output_data_offset = output_frame_index * out_w * out_h * 3 + output_pixel_x * out_w * 3 + output_pixel_y * 3;
        output_data[output_data_offset] = r;
        output_data[output_data_offset + 1] = g;
        output_data[output_data_offset + 2] = b;
    }
}

void process_video(cv::Mat frames[], int frameCount)
{
    cudaError_t err = cudaSuccess;

    // Total number of pixels to process in GPU
    int output_pixels_per_frame = output_width * output_height;

    // Calculate input frames size
    int input_pixels_per_frame = input_width * input_height;

    // Define number of pixels to process per thread
    int availableThreads = num_blocks * num_threads_per_block;

    // Number of iterations
    int nIterations = framesCount / FRAMES_BATCH_SIZE;

    for (int i = 0; i < nIterations; i++)
    {
        // Calculate number of frames to process in this batch (last batch may be smaller)
        int frames_to_process = FRAMES_BATCH_SIZE;
        if (i == nIterations - 1)
        {
            frames_to_process = frameCount - i * FRAMES_BATCH_SIZE;
        }

        // Calculate number of pixels to process in this batch
        int output_pixels_to_process = frames_to_process * output_pixels_per_frame;
        int input_pixels_to_process = frames_to_process * input_pixels_per_frame;

        // Calculate number of pixels per thread
        int pixels_per_thread = output_pixels_to_process / availableThreads;

        // Allocate host memory output frames array
        // Recall that each pixel has 3 channels (RGB)
        size_t input_data_size = input_pixels_to_process * 3 * sizeof(unsigned char);
        size_t output_data_size = output_pixels_to_process * 3 * sizeof(unsigned char);

        unsigned char *d_input_data, *d_output_data;

        // Allocate device memory for input and output frames arrays
        err = cudaMalloc((void **)&d_input_data, input_data_size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device input vector (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Assign input data to host memory
        for (int j = 0; j < frames_to_process; j++)
        {
            int frame_index = i * FRAMES_BATCH_SIZE + j;

            // Copy input frame to host memory
            err = cudaMemcpy(
                d_input_data + j * input_pixels_per_frame * 3,
                frames[frame_index].data,
                input_pixels_per_frame * 3 * sizeof(unsigned char),
                cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(
                    stderr,
                    "Failed to assign input vector frames data at frame index %s (error code %s)!\n",
                    frame_index, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }

        // Allocate device memory for output frames array
        err = cudaMalloc((void **)&d_output_data, output_data_size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device output vector (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Data is assigned and memory allocated, now process pixels in parallel
        process<<<num_blocks, num_threads_per_block>>>(
            d_input_data, d_output_data, frames_to_process,
            input_width, input_height,
            output_width, output_height);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch process kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy frame data from host memory to output frames array

        for (int j = 0; j < frames_to_process; j++)
        {
            int frame_index = i * FRAMES_BATCH_SIZE + j;

            // Copy input frame to host memory
            err = cudaMemcpy(
                frames[frame_index].data,
                h_output_data + j * output_pixels_per_frame * 3,
                output_pixels_per_frame * 3 * sizeof(unsigned char),
                cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                fprintf(
                    stderr,
                    "Failed to assign output vector frames data at frame index %s (error code %s)!\n",
                    frame_index, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }

        // Free device global memory
        err = cudaFree(d_input_data);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaFree(d_output_data);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Reset the device and exit
        // err = cudaDeviceReset();

        // if (err != cudaSuccess)
        // {
        //     fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        //     exit(EXIT_FAILURE);
        // }
    }
}

using namespace std;

int main(int argc, char *argv[])
{
    // Get params size
    if (argc >= 5)
    {
        filename = argv[1];
        output_filename = argv[2];
        num_threads_per_block = atoi(argv[3]);
        num_blocks = atoi(argv[4]);
    }
    else
    {
        filename = (char *)DEFAULT_FILENAME;
        output_filename = (char *)DEFAULT_OUTPUT_FILENAME;
        num_threads_per_block = NUM_THREADS_PER_BLOCK;
        num_blocks = NUM_BLOCKS;
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
