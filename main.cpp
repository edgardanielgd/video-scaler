#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "omp.h"

using namespace std;

cv::Mat process_frame(cv::Mat input)
{
    cv::Mat output;

    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(output, output, cv::Size(7, 7), 1.5, 1.5);

    return output;
}

int main(int argc, char *argv[])
{
    char *filename = argv[1];

    cv::VideoCapture capture(filename);
    cv::VideoWriter writer;

    if (!capture.isOpened())
    {
        return 1;
    }

#pragma omp parallel
    {
        cout << "Number of threads: " << omp_get_num_threads() << endl;
        cv::Mat frame;

// Getting a frame from video should be critical
#pragma omp critical
        {
            capture >> frame;
        }

        if (!frame.empty())
        {
            cv::Mat new_frame = process_frame(frame);
        }
    }

    cout << "Done!" << endl;
}