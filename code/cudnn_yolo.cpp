#include <iostream>
#include <fstream> 
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <sys/time.h>
#include <syslog.h>

using namespace cv;
using namespace dnn;
using namespace std;

#define NUM_FRAMES 1590
#define ESCAPE_KEY (27)
#define SYSTEM_ERROR (-1)
int VRES = 480;
int HRES = 853;

/* Time stamps*/
timespec yolo_start_time,yolo_stop_time;

string streamPath,cfgFile, weightsFile;

// Function to get the output layer names
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        // Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void cudnn_yolo()
{
    // Load names of classes
    string classesFile = "coco.names"; // You can download it from the official YOLO website
    ifstream ifs(classesFile.c_str());
    string line;
    vector<string> classes;
    while (getline(ifs, line)) classes.push_back(line);

    // Load the network
    Net net = readNetFromDarknet(cfgFile, weightsFile);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    VideoCapture capture(streamPath);

    // Load the image
    if (!capture.isOpened())
    {
       exit(SYSTEM_ERROR);
    } 
    namedWindow("YOLO Object Detection", WINDOW_NORMAL);
    
    Mat frame,image;
    int frameNumber = 0, frameCounter = 0;
    int secCounter = 0;

    //generate application start time stamp for fps calculation
    clock_gettime(CLOCK_MONOTONIC,&yolo_start_time);
    double sec_start = (double)(yolo_start_time.tv_sec + (double)(yolo_start_time.tv_nsec)/1000000000);
    double sys_start = sec_start;

    while (frameNumber < NUM_FRAMES) 
    {
        //count number of frames
        frameNumber++;
        frameCounter++; 

        //get the image from the stream
        capture.read(image);

        // Run forward pass to get output of the output layers
        clock_gettime(CLOCK_MONOTONIC,&yolo_start_time);
        double time1 = (double)(yolo_start_time.tv_sec + (double)(yolo_start_time.tv_nsec)/1000000000);

        resize(image,frame,Size(HRES,VRES),INTER_LINEAR);

        // Create a 4D blob from a frame
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

        // Set the input to the network
        net.setInput(blob);

        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        float confThreshold = 0.7;
        float nmsThreshold = 0.4;
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        int count = 0;
        int minWidth = 50;
        int minHeight = 50;
        

        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                count++;
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    // if((width < minWidth) || (height < minHeight))
                    // {
                    //     continue;
                    // }
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
        // cout<<count<<endl;

        // Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        // Draw the bounding boxes on the frame
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            int classId = classIds[idx];
            string label = format("%.2f", confidences[idx]);
            if (!classes.empty())
            {
                CV_Assert(classId < (int)classes.size());
                label = classes[classId] + ":" + label;
            }
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        clock_gettime(CLOCK_MONOTONIC,&yolo_stop_time);
        double time2 = (double)(yolo_stop_time.tv_sec + (double)(yolo_stop_time.tv_nsec)/1000000000);
        // printf("\nRuntime: %lf",(time2-time1));

        //get the total time elapsed and frames processed
        double delta = time2 - sys_start;

        //log the framerate every second
        if((int)delta > secCounter)
        {
            secCounter++;
            double freq = (double)frameCounter/(time2 - sec_start) ; 
            // printf("\nfps: %lf",freq);
            syslog(LOG_INFO, "\nYolo Thread ... time elapsed:%lf fps: %lf",delta,freq);
            frameCounter = 0;
            clock_gettime(CLOCK_MONOTONIC,&yolo_start_time);
            sec_start = (double)(yolo_start_time.tv_sec + (double)(yolo_start_time.tv_nsec)/1000000000);
            
        }

        // Display the result
        imshow("YOLO Object Detection", frame);
        char winInput = waitKey(1);
        if (winInput == ESCAPE_KEY)
        {
            break;
        }
    }
}

int main(int argc, char** argv)
{
    // Check for proper usage
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <path_to_video> <path_to_yolov4cfg> <path_to_yolov4.weights>" << endl;
        return -1;
    }

    openlog("Logs",LOG_PID, LOG_USER);

    streamPath = argv[1];
    cfgFile = argv[2];
    weightsFile = argv[3];

    cudnn_yolo();



    return 0;
}
