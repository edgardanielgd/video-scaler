#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <chrono>
#include "omp.h"

#define NUM_THREADS 4
#define DEFAULT_FILENAME "video2.mp4"
#define DEFAULT_OUTPUT_FILENAME "output.mp4"
#define DEFAULT_OUTPUT_WIDTH 640
#define DEFAULT_OUTPUT_HEIGHT 360

#define NEAREST_NEIGHBOR 0
#define BILINEAR_INTERPOLATION 1
#define BICUBIC_INTERPOLATION 2

using namespace std;

char *filename;
char *output_filename;
int input_width, output_width;
int input_height, output_height;
int num_threads;
int method;

cv::Mat process_frame(cv::Mat input)
{
    cv::Mat output(output_height, output_width, CV_8UC3);

    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            switch (method)
            {
            case NEAREST_NEIGHBOR:
            {
                int x = (int)(i * (input_height / output_height));
                int y = (int)(j * (input_width / output_width));

                output.at<cv::Vec3b>(i, j) = input.at<cv::Vec3b>(x, y);
                break;
            }

            case BILINEAR_INTERPOLATION: // Same than INTER_LINEAL
            {
                int x = (int)(i * (input_height / output_height));
                int y = (int)(j * (input_width / output_width));

                // Average 2x2 neighbors values
                int r = 0, g = 0, b = 0;
                int count = 0;
                for (int xn = floor(x); xn <= min((int)ceil(x), input_height - 1); xn++)
                {
                    for (int yn = floor(y); yn <= min((int)ceil(y), input_width - 1); yn++)
                    {
                        r += input.at<cv::Vec3b>(xn, yn)[0];
                        g += input.at<cv::Vec3b>(xn, yn)[1];
                        b += input.at<cv::Vec3b>(xn, yn)[2];

                        count++;
                    }
                }

                output.at<cv::Vec3b>(i, j) = cv::Vec3b(r, g, b) / count;
                break;
            }

            default:
            {
                output.at<cv::Vec3b>(i, j) = input.at<cv::Vec3b>(0, 0);
            }
            }
        }
    }

    return output;
}

int main(int argc, char *argv[])
{
    // Get params size
    if (argc >= 4)
    {
        filename = argv[1];
        output_filename = argv[2];
        num_threads = atoi(argv[3]);
    }
    else
    {
        filename = (char *)DEFAULT_FILENAME;
        output_filename = (char *)DEFAULT_OUTPUT_FILENAME;
        num_threads = NUM_THREADS;
    }
    if (argc >= 6)
    {
        output_width = atoi(argv[4]);
        output_height = atoi(argv[5]);
    }
    else
    {
        output_width = DEFAULT_OUTPUT_WIDTH;
        output_height = DEFAULT_OUTPUT_HEIGHT;
    }

    if (argc >= 7)
    {
        method = atoi(argv[6]);
    }
    else
    {
        method = BILINEAR_INTERPOLATION;
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

    cout << "Video metadata: (" << num_threads << ")" << endl;
    cout << "  - Frame width: " << input_width << endl;
    cout << "  - Frame height: " << input_height << endl;
    cout << "  - Frame count: " << frame_count << endl;
    cout << "  - FPS: " << fps << endl;

    cout << "Output video metadata:" << endl;
    cout << "  - Frame width: " << output_width << endl;
    cout << "  - Frame height: " << output_height << endl;

    assert(input_width >= output_width && input_height >= output_height);

    // Copying video frame by frame into video writter
    cv::VideoWriter writter("output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(output_width, output_height));

    // We'll write output video frame by frame in parallel
    cv::Mat frames[frame_count];

    cout << "Reading video frames..." << endl;
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        capture >> frames[frame_number];
    }

    capture.release();

    omp_set_num_threads(NUM_THREADS);

    auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        cv::Mat frame = frames[frame_number];

        // Here frame is processed in parallel
        frames[frame_number] = process_frame(frame);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Threads: " << num_threads << " Execution Time: " << duration.count() << endl;

    // Joining all frames into a single video
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        writter << frames[frame_number];
    }

    writter.release();
}