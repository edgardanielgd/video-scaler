#include <mpi.h>
#include <stdio.h>
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

#define DEFAULT_FILENAME "video1.mp4"
#define DEFAULT_OUTPUT_FILENAME "output"
#define OUTPUT_WIDTH 480
#define OUTPUT_HEIGHT 360
#define THREADS_PER_PROCESS 4

using namespace std;

cv::Mat process_frame(cv::Mat input, int input_width, int input_height)
{
    cv::Mat output(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);

    for (int i = 0; i < OUTPUT_HEIGHT; i++)
    {
        for (int j = 0; j < OUTPUT_WIDTH; j++)
        {
            int x = (int)(i * (input_height / OUTPUT_HEIGHT));
            int y = (int)(j * (input_width / OUTPUT_WIDTH));

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
        }
    }

    return output;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cv::VideoCapture capture(DEFAULT_FILENAME);

    if (!capture.isOpened())
    {
        std::cout << "Error when reading video file" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int input_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int input_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);

    int fps = capture.get(cv::CAP_PROP_FPS);
    int video_time = frame_count / fps;

    // Read frames from file only at the root process
    int frames_to_process = frame_count / world_size;
    int frames_offset = frames_to_process * world_rank;

    // Last process will process the remaining frames
    if (world_rank == world_size - 1)
    {
        frames_to_process += frame_count % world_size;
    }

    cv::Mat frames[frames_to_process];

    // Set capture to the first frame to process
    capture.set(cv::CAP_PROP_POS_FRAMES, frames_offset);

    cout << "Starts capturing frames at " << frames_offset << " (" << frames_to_process << ")" << endl;

    // Process frames in parallel with OpenMP
    omp_set_num_threads(THREADS_PER_PROCESS);
    // #pragma omp parallel for
    for (
        int frame_number = 0;
        frame_number < frames_to_process;
        frame_number++)
    {
        cv::Mat frame;
        capture >> frame;

        // Here frame is processed in parallel
        frames[frame_number] = process_frame(frame, input_width, input_height);
    }

    capture.release();

    cout << "Ends capturing frames at " << frames_offset << " (" << frames_to_process << ")" << endl;

    // Save frames to file
    string output_filename = DEFAULT_OUTPUT_FILENAME + to_string(world_rank) + ".mp4";
    cout << "Output file: " << output_filename << endl;

    cv::VideoWriter writter(output_filename,
                            cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                            fps,
                            cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

    cout << "Starts writing frames at " << frames_offset << " (" << frames_to_process << ")" << endl;
    for (int frame_number = 0; frame_number < frames_to_process; frame_number++)
    {
        writter << frames[frame_number];
    }

    cout << "Ends writing frames at " << frames_offset << " (" << frames_to_process << ")" << endl;

    writter.release();

    // Wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // Merge all output files
        string command = "ffmpeg";

        for (int i = 0; i < world_size; i++)
        {
            string output_part_filename = DEFAULT_OUTPUT_FILENAME + to_string(i) + ".mp4";
            command += string(" -i ") + output_part_filename;
        }

        string output_filename = DEFAULT_OUTPUT_FILENAME + string(".mp4");

        command += string(" -filter_complex ") + "\"" + string("concat=n=") + to_string(world_size) + string(":v=1:a=0") + "\"" + string(" -y ") + output_filename;

        cout << "Merging output files" << endl;
        cout << command << endl;

        system(command.c_str());
    }

    MPI_Finalize();

    return 0;
}