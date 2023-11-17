#include <mpi.h>
#include <stdio.h>
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

#define DEFAULT_FILENAME "video1.mp4"
#define DEFAULT_OUTPUT_FILENAME "output.mp4"
#define OUTPUT_WIDTH 480
#define OUTPUT_HEIGHT 360

using namespace std;

char *process_frame(char *frame, int input_width, int input_height)
{
    char *output_frame = (char *)malloc(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3 * sizeof(char));
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
                    char *pixel = frame + xn * input_width * 3 + yn * 3;
                    r += pixel[0];
                    g += pixel[1];
                    b += pixel[2];

                    count++;
                }
            }

            char *output_pixel = output_frame + i * OUTPUT_WIDTH * 3 + j * 3;
            output_pixel[0] = (char)(r / count);
            output_pixel[1] = (char)(g / count);
            output_pixel[2] = (char)(b / count);
        }
    }

    return output_frame;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int input_width = 0;
    int input_height = 0;
    int frame_count = 0;

    // Read frames from file only at the root process
    if (world_rank == 0)
    {
        cv::VideoCapture capture(DEFAULT_FILENAME);

        if (!capture.isOpened())
        {
            std::cout << "Error when reading video file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        input_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        input_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);

        int fps = capture.get(cv::CAP_PROP_FPS);
        int video_time = frame_count / fps;

        cout << "Video metadata: " << endl;
        cout << "  - Frame width: " << input_width << endl;
        cout << "  - Frame height: " << input_height << endl;
        cout << "  - Frame count: " << frame_count << endl;
        cout << "  - FPS: " << fps << endl;

        cout << "Output video metadata:" << endl;
        cout << "  - Frame width: " << OUTPUT_WIDTH << endl;
        cout << "  - Frame height: " << OUTPUT_HEIGHT << endl;

        cout << "Input file: " << DEFAULT_FILENAME << endl;
        cout << "Output file: " << DEFAULT_OUTPUT_FILENAME << endl;

        assert(input_width >= OUTPUT_WIDTH && input_height >= OUTPUT_HEIGHT);

        // Send metadata to all procsses first
        int res = MPI_Bcast(&input_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        res = MPI_Bcast(&input_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        res = MPI_Bcast(&frame_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Start reading video frames
        cv::Mat frame;
        for (int frame_index = 0; frame_index < frame_count; ++frame_index)
        {
            capture.read(frame);

            int target_rank = frame_index % (world_size - 1) + 1;

            // Distribute frames to other ranks
            MPI_Send(
                frame.data,
                input_width * input_height * 3,
                MPI_INT,
                target_rank,
                0,
                MPI_COMM_WORLD);
        }

        cout << "Finished sending frames" << endl;

        capture.release();

        cv::VideoWriter writter(
            DEFAULT_OUTPUT_FILENAME,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps, cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

        // Wait for incoming messages
        while (MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
        {
            // Receive processed frame from any rank
            char *output_frame = (char *)malloc(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3 * sizeof(char));
            MPI_Recv(
                output_frame,
                OUTPUT_WIDTH * OUTPUT_HEIGHT * 3,
                MPI_BYTE,
                MPI_ANY_SOURCE,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

            // Write processed frame to output file
            cv::Mat output_mat(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3, output_frame);
            writter.write(output_mat);
        }

        writter.release();
    }
    else
    {
        // Read metadata from root process
        cout << "Waiting for metadata" << endl;

        // Get metadata from root process
        int res = MPI_Bcast(&input_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        res = MPI_Bcast(&input_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        res = MPI_Bcast(&frame_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (res != MPI_SUCCESS)
        {
            cout << "Error when broadcasting metadata" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cout << "Gotten metadata: "
             << "input_width = " << input_width << ", "
             << "input_height = " << input_height << ", "
             << "frame_count = " << frame_count << endl;

        // While there are still frames to process
        while (MPI_Probe(0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
        {
            int test_data[input_width * input_height * 3];
            MPI_Recv(
                test_data,
                input_width * input_height * 3,
                MPI_INT,
                0,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            // Receive frame from root process
            char *frame = (char *)malloc(input_width * input_height * 3 * sizeof(char));
            MPI_Recv(
                frame,
                input_width * input_height * 3,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

            // Process frame
            char *output_frame = process_frame(frame, input_width, input_height);

            // Send processed frame back to root process
            MPI_Send(
                output_frame,
                OUTPUT_WIDTH * OUTPUT_HEIGHT * 3,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}