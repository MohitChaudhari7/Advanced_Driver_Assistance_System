/*****************************************************************
 * Includes
*******************************************************************/
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <string>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <string>
#include <syslog.h>
#include <getopt.h>
#include <signal.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "opencv2/cudafilters.hpp"


#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)
#define NUM_FRAMES 2000
#define NUM_THREADS (2)

using namespace cv;
using namespace std;

/* Input stream, resize resolution */
// int VRES = 750;
// int HRES = 1200;
int VRES = 480;
int HRES = 854;

int show_flag = 0, exit_flag = 0;
string output_file = "output.avi";

/* Time stamps*/
timespec lane_start_time,lane_stop_time,sched_start_time,haar_start_time,haar_stop_time;

/* Image*/
VideoCapture stream;
Mat image;
/*haar rectangles*/
pthread_mutex_t RectMutex, ImgMutex;
vector<cv::Rect> found;

typedef struct
{
    int threadIdx;

}threadParams_t;

pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
struct sched_param rt_param[NUM_THREADS];
struct sched_param main_param;
pthread_attr_t main_attr;
int rt_max_prio, rt_min_prio;
pid_t mainpid;

/*Semaphores*/
sem_t laneSem, haarSem;
pthread_mutex_t imgMutex;

Vec4i convert_to_cartesian(Vec2f line_segment) 
{
    int height = VRES;
    int width = HRES;
    int y1 = height; 
    int y2 = static_cast<int>(y1 * 1 / 2); 

    float slope = line_segment[0];
    float intercept = line_segment[1];

    // convert line to cartesian and keep the points within the frame
    int x1 = max(-width, min(2 * width, static_cast<int>((y1 - intercept) / slope)));
    int x2 = max(-width, min(2 * width, static_cast<int>((y2 - intercept) / slope)));

    return Vec4i(x1, y1, x2, y2);
}

vector<Vec4i> average_line_generator(vector<Vec4i> detected_lines) 
{
    
    vector<Vec4i> average_lines;
    if (detected_lines.empty()) 
    {
        return average_lines;
    }

    int height = VRES;
    int width = HRES;
    int leftCount = 0, rightCount = 0;

    float mid = float(width)* 0.4;
    Vec2f left_avg_line(0, 0),right_avg_line(0, 0);

    for (size_t i = 0; i < detected_lines.size(); i++) 
    {
        Vec4i line_segment = detected_lines[i];
        int x1 = line_segment[0], y1 = line_segment[1], x2 = line_segment[2], y2 = line_segment[3];

        //skip vertical lines as airthmetic error while calculating slope
        if (x1 == x2) 
        {
            continue;
        }
        float slope = float(y2-y1)/float(x2-x1);
        float intercept= float(y1)-(slope*float(x1));

        if(abs(slope) < 0.5)
        {
            continue;
        }
        if (slope < 0) 
        {
            if (x1 < mid && x2 < mid) 
            {
                leftCount++;
                left_avg_line[0] += slope;
                left_avg_line[1] += intercept;
            }
        } else 
        {
            if (x1 > mid && x2 > mid) 
            {
                rightCount++;
                right_avg_line[0] += slope;
                right_avg_line[1] += intercept;
            }
        }
    }

    if (leftCount) 
    {
            
        left_avg_line[0] /= static_cast<float>(leftCount);
        left_avg_line[1] /= static_cast<float>(leftCount);
        average_lines.push_back(convert_to_cartesian(left_avg_line));
    }

    if (rightCount) 
    {

        right_avg_line[0] /= static_cast<float>(rightCount);
        right_avg_line[1] /= static_cast<float>(rightCount);
        average_lines.push_back(convert_to_cartesian(right_avg_line));
    }
    return average_lines;
}

cuda::GpuMat CUDA_region_of_interest(cuda::GpuMat gpu_edges) 
{
    Mat edges;
    gpu_edges.download(edges);
    int height = edges.rows;
    int width = edges.cols;
    Mat mask = Mat::zeros(edges.size(), edges.type());

    // Define the region of interest polygon
    vector<Point> vertices;
    int buf_val = 200 * VRES / 1080;
    int height_val = 300 * VRES / 1080;
    vertices.push_back(Point(width*1/4 + buf_val, height * 1 / 4 + height_val));
    vertices.push_back(Point(width*3/4 - (2.5*buf_val), height * 1 / 4 + height_val));
    vertices.push_back(Point(width*3/4 , height* 3/4 + buf_val));
    vertices.push_back(Point(width*1/4 - (1.5*buf_val), height* 3/4 + buf_val));
    vector<vector<Point>> polygons = {vertices};

    // Fill the polygon with white color (255)
    fillPoly(mask, polygons, Scalar(255));

    // Apply the mask to the edges image
    Mat cropped_edges;
    bitwise_and(edges, mask, cropped_edges);
    gpu_edges.upload(cropped_edges);
    if(show_flag)
    {
        imshow("Roi",cropped_edges);
    }
    return gpu_edges;
}

void *HaarCar(void *threadp)
{
    // Load the car data
    Ptr<cuda::CascadeClassifier> car_data = cuda::CascadeClassifier::create("cars.xml");
    if (car_data.empty()) {
        cout << "Error loading cars.xml" << endl;
        exit(-1);
    }
    car_data->setScaleFactor(1.1);
    car_data->setMinNeighbors(10);
    car_data->setMinObjectSize(cv::Size(70,70));

    // Allocate memory for GPU frames
    cuda::GpuMat gpu_frame, gpu_gray, gpu_blur;

    int frameNumber = 0, frameCounter = 0;
    int secCounter = 0;

    //generate application start time stamp for fps calculation
    clock_gettime(CLOCK_MONOTONIC,&haar_start_time);
    double sec_start = (double)(haar_start_time.tv_sec + (double)(haar_start_time.tv_nsec)/1000000000);
    double sys_start = sec_start;

    while (!exit_flag) 
    {
        
        sem_wait(&haarSem);    

        frameNumber++;
        frameCounter++;

        // Read the image
        // Mat img = image;

        clock_gettime(CLOCK_MONOTONIC,&haar_start_time);
        double time1 = (double)(haar_start_time.tv_sec + (double)(haar_start_time.tv_nsec)/1000000000);

        //resize the image to required resolution
        Mat img;
        pthread_mutex_lock(&ImgMutex);
        resize(image,img,Size(HRES,VRES),INTER_LINEAR);
        pthread_mutex_unlock(&ImgMutex);

        // Upload frame to GPU
        gpu_frame.upload(img);

        // Convert to grayscale
        cuda::cvtColor(gpu_frame, gpu_gray, COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(gpu_gray.type(), gpu_blur.type(), Size(3, 3), 0);
        gaussian_filter->apply(gpu_gray, gpu_blur);

        // Detect cars
        cuda::GpuMat gpu_found;
        car_data->detectMultiScale(gpu_blur, gpu_found);

        // Download results from GPU to CPU
        // vector<Rect> found;
        car_data->convert(gpu_found, found);

        // // Draw rectangles around detected cars
        // for (size_t i = 0; i < found.size(); i++) 
        // {
        //     rectangle(img, found[i], Scalar(0, 255, 0), 5);
        // }

        // if(show_flag)
        // {
            
        //     imshow( "Final", img );
        // }
        // char wait_char = waitKey(1);
        // if( wait_char == ESCAPE_KEY )
        // {
        //     printf("\nExit\n"); 
        //     break;
        // }

        clock_gettime(CLOCK_MONOTONIC,&haar_stop_time);
        double time2 = (double)(haar_stop_time.tv_sec + (double)(haar_stop_time.tv_nsec)/1000000000);
        // printf("\nRuntime: %lf",(time2-time1));

        //get the total time elapsed and frames processed
        double delta = time2 - sys_start;

        //log the framerate every second
        if((int)delta > secCounter)
        {
            secCounter++;
            double freq = (double)frameCounter/(time2 - sec_start) ; 
            // printf("\nfps: %lf",freq);
            syslog(LOG_INFO, "\nHaar Thread ... time elapsed:%lf fps: %lf",delta,freq);
            frameCounter = 0;
            clock_gettime(CLOCK_MONOTONIC,&haar_start_time);
            sec_start = (double)(haar_start_time.tv_sec + (double)(haar_start_time.tv_nsec)/1000000000);
            
        }
    }
    return threadp;

}

void *LaneDetection(void *threadp)
{

    if(show_flag)
    {
        // namedWindow( "Original",WINDOW_NORMAL );
        namedWindow( "Hough",WINDOW_NORMAL );
        // namedWindow( "Roi",WINDOW_NORMAL );
        namedWindow( "Final",WINDOW_NORMAL );
    }
 
    int frame_width = stream.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = stream.get(CAP_PROP_FRAME_HEIGHT);

    // VideoWriter video(output_file,VideoWriter::fourcc('M','J','P','G'),10, Size(frame_width,frame_height),true);
    
    // Transform display window
    vector<Vec4i> detected_lines;

    Mat frame_blur, frame_gray, canny_output, canny_roi;

    Mat frame,hough;

    int kernel_size = 3;

    int frameNumber = 0, frameCounter = 0;
    int secCounter = 0;

    //generate application start time stamp for fps calculation
    clock_gettime(CLOCK_MONOTONIC,&lane_start_time);
    double sec_start = (double)(lane_start_time.tv_sec + (double)(lane_start_time.tv_nsec)/1000000000);
    double sys_start = sec_start;

    while (frameNumber < NUM_FRAMES) 
    {
        //resize the image to required resolution
        pthread_mutex_lock(&ImgMutex);
        stream.read(image);
        resize(image,frame,Size(HRES,VRES),INTER_LINEAR);
        pthread_mutex_unlock(&ImgMutex);

        sem_wait(&laneSem);
        // stream.read(image);

        clock_gettime(CLOCK_MONOTONIC,&lane_start_time);
        double time1 = (double)(lane_start_time.tv_sec + (double)(lane_start_time.tv_nsec)/1000000000);

        //count number of frames
        frameNumber++;
        frameCounter++; 

        frame.copyTo( hough, frame);

        cuda::GpuMat gpu_src,gpu_dst,gpu_gray,gpu_edge,gpu_roi,gpu_lines;

        //upload the frame to the gpu device
        gpu_src.upload(frame);

        //convert to grayscale
        //cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        cuda::cvtColor(gpu_src, gpu_gray, COLOR_BGR2GRAY);

        //blurr the image
        cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(gpu_gray.type(), gpu_dst.type(), Size(5, 5), 0);
        gaussian_filter->apply(gpu_gray, gpu_dst);

        // Canny detector
        cv::Ptr<cv::cuda::CannyEdgeDetector> gpuEdgeDetector =cv::cuda::createCannyEdgeDetector(20.0,70.0);
        gpuEdgeDetector->detect(gpu_dst,gpu_edge);

        //remove parts of the image which are unnecessary
        gpu_roi = CUDA_region_of_interest(gpu_edge);
    
        //hough transform
        //HoughLinesP(canny_roi, detected_lines, 1, CV_PI / 180, 50, 50, 10);
        cv::Ptr<cv::cuda::HoughSegmentDetector> hough_detector =cv::cuda::createHoughSegmentDetector(1, CV_PI / 180, 110, 50, 20);
        hough_detector->detect(gpu_roi, gpu_lines);
        
        if (!gpu_lines.empty()) 
        {
            detected_lines.resize(gpu_lines.cols);
            cv::Mat hough_lines_mat(1, gpu_lines.cols, CV_32SC4, &detected_lines[0]);
            gpu_lines.download(hough_lines_mat);
        }

        //get the average lines 
        vector<Vec4i> average_lines= average_line_generator(detected_lines);

        //draw the average left and right lane lines
        for (size_t i = 0; i < average_lines.size(); i++) 
        {
            Vec4i line_segment = average_lines[i];
            line(frame, Point(line_segment[0], line_segment[1]), Point(line_segment[2], line_segment[3]), Scalar(255,0,0), 3, LINE_AA);

        }

        //draw the haar rectangles
        pthread_mutex_lock(&RectMutex);
        for (size_t i = 0; i < found.size(); i++) 
        {
            rectangle(frame, found[i], cv::Scalar(0, 255, 0), 5);
        }
        pthread_mutex_unlock(&RectMutex);

        if(show_flag)
        {
            //drawing the lines on the original image            
            for( size_t i = 0; i < detected_lines.size(); i++ )
            {
                Vec4i line_segment = detected_lines[i];
                line( hough, Point(line_segment[0],line_segment[1]), Point(line_segment[2], line_segment[3]), Scalar(255,0,0), 3, LINE_AA);
            }
            imshow( "Hough", hough );
            imshow( "Final", frame );
        }

        // video.write(frame);
        // generate file path based on frame number
        char frame_no[10];
        char format[50] = ".jpg";
        frame_no[0] = (int)(frameNumber/1000)%10 + '0';
        frame_no[1] = (int)(frameNumber/100)%10 + '0';
        frame_no[2] = (int)(frameNumber/10)%10 + '0';
        frame_no[3] = frameNumber%10 + '0';
        frame_no[4] = '\0';
        //cout<<frame_no<<endl;
        strcat(frame_no,format);
        char write_base[50] = "/home/mohit/ECV/final/cuda_haar_car/frame";
        strcat(write_base,frame_no);

        //save the image
        imwrite(write_base,frame);

        char wait_char = waitKey(1);
        if( wait_char == ESCAPE_KEY )
        {
            printf("\nExit\n"); 
            break;
        }
        

        clock_gettime(CLOCK_MONOTONIC,&lane_stop_time);
        double time2 = (double)(lane_stop_time.tv_sec + (double)(lane_stop_time.tv_nsec)/1000000000);
        // printf("\nRuntime: %lf",(time2-time1));

        //get the total time elapsed and frames processed
        double delta = time2 - sys_start;

        //log the framerate every second
        if((int)delta > secCounter)
        {
            secCounter++;
            double freq = (double)frameCounter/(time2 - sec_start) ; 
            // printf("\nfps: %lf",freq);
            syslog(LOG_INFO, "\nLane Thread ... time elapsed:%lf fps: %lf",delta,freq);
            frameCounter = 0;
            clock_gettime(CLOCK_MONOTONIC,&lane_start_time);
            sec_start = (double)(lane_start_time.tv_sec + (double)(lane_start_time.tv_nsec)/1000000000);
            
        }
    }
    exit_flag = 1;

    if(show_flag)
    {
        // destroyWindow("Original");
        destroyWindow("Hough");
        // destroyWindow("Roi");
        destroyWindow("Final");
    }

    return threadp;
}


void rt_Init(void)
{
    int rc, scope;
    int i = 0;
    cpu_set_t threadcpu[2];

    CPU_ZERO(&threadcpu[0]);   //No cpu is selected
    CPU_SET(2, &threadcpu[0]);   //CPU 2 is selected to execute the thread   
    CPU_ZERO(&threadcpu[1]);   //No cpu is selected
    CPU_SET(3, &threadcpu[1]);   //CPU 3 is selected to execute the thread      
\
    mainpid=getpid();
   
    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param); //Getting parameters of mainpid. Parameters are stored in &main_param
    main_param.sched_priority=rt_max_prio; //Setting priority of FIFO to max
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param); // getpid() -> set the policy for the pid returned 
                                                              //main_param : structure containing the scehduling parameters to be set
    if(rc < 0) perror("main_param");

    printf("SCHED FIFO\n");

    pthread_attr_getscope(&main_attr, &scope); // Retrieves contention scope

    if(scope == PTHREAD_SCOPE_SYSTEM)
    {
        printf("PTHREAD SCOPE SYSTEM\n");
    }
    else if (scope == PTHREAD_SCOPE_PROCESS)
    {
        printf("PTHREAD SCOPE PROCESS\n");
    }
    else
    {
        printf("PTHREAD SCOPE UNKNOWN\n");
    }

   printf("rt_max_prio=%d\n", rt_max_prio);
   printf("rt_min_prio=%d\n", rt_min_prio);

    for(i=0; i < NUM_THREADS; i++)
    {
        rc = pthread_attr_init(&rt_sched_attr[i]);
        rc = pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        rc = pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO); /*Setting the thread policy to be FIFO*/
        rc = pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu[i]); /*Selecting CPU 1 - the thread is confined to running only on CPU1*/
        // Setting the priority for increment thread to max and decrement thread to just below max

        pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

        pthread_attr_setschedparam(&main_attr, &main_param);
        threadParams[i].threadIdx=i;
    }
    rt_param[0].sched_priority = rt_max_prio-2; //lane detection has higher priority
    rt_param[1].sched_priority = rt_max_prio-3; //haar detection has lower priority

    if (rc)
    {
        printf("ERROR: sched_setscheduler rc is %d\n", rc);
        perror("sched_setscheduler");
        exit(SYSTEM_ERROR);
    }
    pthread_create(&threads[0],   // pointer to thread descriptor
                    &rt_sched_attr[0],     // use default attributes
                    LaneDetection, // thread function entry point
                    (void *)&(threadParams[0]) // parameters to pass in
                    );

    pthread_create(&threads[1],   // pointer to thread descriptor
                    &rt_sched_attr[1],     // use default attributes
                    HaarCar, // thread function entry point
                    (void *)&(threadParams[1]) // parameters to pass in
                    );
}


void rt_scheduler(int signum)
{
    timespec end_time; 
    clock_gettime(CLOCK_REALTIME, &sched_start_time);
    static int timeCount = 0;
    timeCount++;
    if(timeCount % 4 == 0)
    {
        // pthread_mutex_lock(&ImgMutex);
        // stream.read(image);
        // pthread_mutex_unlock(&ImgMutex);
        sem_post(&laneSem);
    }
    if(timeCount % 5 == 0)
    {
        sem_post(&haarSem);
    }
    
    struct itimerval timer;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 10000;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;

    setitimer(ITIMER_REAL, &timer, NULL);
    clock_gettime(CLOCK_REALTIME, &end_time);
}

int main(int argc, char** argv)
{
    // Check command line arguments
    if(argc < 2) 
    {
        printf("Usage: laneDetection <input-file>\n");
        exit(-1);
    }

    //stream setup
    for (int i = 2; i < argc; ++i) 
    {
        std::string arg = argv[i];

        if (arg == "-s") {
            show_flag = 1;
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    stream.open(argv[1]);

    if (!stream.isOpened())
    {
        exit(SYSTEM_ERROR);
    } 

    openlog("Logs",LOG_PID, LOG_USER);

    sem_init(&laneSem,0,0);
    sem_init(&haarSem,0,0);
    pthread_mutex_init(&ImgMutex, NULL);
    pthread_mutex_init(&RectMutex, NULL);
    rt_Init();

    usleep(2000000);
    struct sigaction sa;
    struct itimerval timer;

    sa.sa_handler = rt_scheduler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);

    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 10000;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;

    setitimer(ITIMER_REAL, &timer, NULL);

    for(int i=0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

}