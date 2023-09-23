#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "omp.h"

#define NUM_THREADS 4
#define DEFAULT_FILENAME "video2.mp4"
#define DEFAULT_OUTPUT_FILENAME "output.mp4"
#define DEFAULT_OUTPUT_WIDTH 640
#define DEFAULT_OUTPUT_HEIGHT 480

#define NEAREST_NEIGHBOR 0
#define BILINEAR_INTERPOLATION 1
#define BICUBIC_INTERPOLATION 2

using namespace std;

char *filename;
char *output_filename;
int input_width, output_width;
int input_height, output_height;
int method = BILINEAR_INTERPOLATION;

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

                // cout << x << " " << y << endl;
                output.at<cv::Vec3b>(i, j) = input.at<cv::Vec3b>(x, y);
                break;
            }

            case BILINEAR_INTERPOLATION:
            {
                // Average 2x2 neighbors values
                int r = 0, g = 0, b = 0;
                for (int x = max(0, i - 1); x <= min(input_height - 1, i + 1); x++)
                {
                    for (int y = max(0, j - 1); y <= min(input_width - 1, j + 1); y++)
                    {
                        r += input.at<cv::Vec3b>(x, y)[0] / 4;
                        g += input.at<cv::Vec3b>(x, y)[1] / 4;
                        b += input.at<cv::Vec3b>(x, y)[2] / 4;
                    }
                }

                output.at<cv::Vec3b>(i, j) = cv::Vec3b(r, g, b);
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
    if (argc == 1)
    {
        filename = (char *)DEFAULT_FILENAME;
        output_filename = (char *)DEFAULT_OUTPUT_FILENAME;
        output_width = DEFAULT_OUTPUT_WIDTH;
        output_height = DEFAULT_OUTPUT_HEIGHT;
    }
    else if (argc == 5)
    {
        filename = argv[1];
        output_filename = argv[2];
        output_width = atoi(argv[3]);
        output_height = atoi(argv[4]);
    }
    else
    {
        cout << "Usage: " << argv[0] << " <input_video> <output_video> <output_width> <output_height>" << endl;
        return 1;
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

    cout << "Video metadata:" << endl;
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
    cv::Mat output_frames[frame_count];

    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        cv::Mat frame;

        // Reading a specific frame shouldn't be done in parallel since indexes are not consecutive
#pragma omp critical
        {
            // Set frame position
            capture.set(cv::CAP_PROP_POS_FRAMES, frame_number);

            capture >> frame;
        }

        // Here frame is processed in parallel
        output_frames[frame_number] = process_frame(frame);
    }

    // Joining all frames into a single video
    for (int frame_number = 0; frame_number < frame_count; frame_number++)
    {
        writter << output_frames[frame_number];
    }

    cout << "Done!" << endl;

    // Release video capture and writter
    capture.release();
    writter.release();
}