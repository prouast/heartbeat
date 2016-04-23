//
//  main.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 29/02/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "RPPGSimple.hpp"
#include "RPPGSimpleBox.hpp"
#include "RPPGDetailed.hpp"
#include "opencv.hpp"
#include "FFmpegDecoder.hpp"
#include "FFmpegEncoder.hpp"

using namespace std;
using namespace cv;

#define FACE_CLASSIFIER_PATH "/Users/prouast/Developer/Xcode/Heartbeat/res/haarcascade_frontalface_alt.xml"
#define LEFT_EYE_CLASSIFIER_PATH "/Users/prouast/Developer/Xcode/Heartbeat/res/haarcascade_lefteye_2splits.xml"
#define RIGHT_EYE_CLASSIFIER_PATH "/Users/prouast/Developer/Xcode/Heartbeat/res/haarcascade_righteye_2splits.xml"
#define POSE_ESTIMATOR_PATH "/Users/prouast/Developer/Xcode/Heartbeat/res/shape_predictor_68_face_landmarks.dat"
#define LOG_FILE_PATH "/Users/prouast/Developer/R/Heartrate/Data/"

int main(int argc, const char * argv[]) {
    
    if (argc < 2) {
        cout << "Please provide path to video as argument!" << endl;
        return 0;
    }
    
    FFmpegDecoder decoder;
    std::string f(argv[1]);
    
    cout << "Working with file " << f << endl;
    
    if (decoder.OpenFile(f)) {
        
        // Configure logfile path
        std::ostringstream filepath;
        filepath << LOG_FILE_PATH << "Android_ffmpeg";
        const string LOG_FILE_NAME = filepath.str();
        
        // Load video information
        const int WIDTH = decoder.GetWidth();
        const int HEIGHT = decoder.GetHeight();
        const int FRAME_COUNT = 10000; // TODO PROPERLY!
        const double FPS = decoder.GetFPS();
        const double TIME_BASE = decoder.GetTimeBase();
        
        // Print video information
        cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
        cout << "FPS: " << FPS << endl;
        cout << "FRAME COUNT: " << FRAME_COUNT << endl;
        cout << "TIME BASE: " << TIME_BASE << endl;
        cout << "START TIME: " << decoder.GetStartTime() << endl;
        
        // allocate and init a re-usable frame
        AVFrame *decoded;
        decoded = av_frame_alloc();
        if (!decoded) {
            fprintf(stderr, "Could not allocate video frame\n");
            return false;
        }
        
        // Set up controller for simple algorithm
        RPPGSimple simple = RPPGSimple(WIDTH, HEIGHT,
                                       TIME_BASE, 1, 1,
                                       LOG_FILE_NAME,
                                       FACE_CLASSIFIER_PATH,
                                       LEFT_EYE_CLASSIFIER_PATH,
                                       RIGHT_EYE_CLASSIFIER_PATH,
                                       true, true);
        
        // Set up controller for simple box algorithm
        RPPGSimpleBox simpleBox = RPPGSimpleBox();
        simpleBox.load(WIDTH, HEIGHT, TIME_BASE,
                       FACE_CLASSIFIER_PATH,
                       LEFT_EYE_CLASSIFIER_PATH,
                       RIGHT_EYE_CLASSIFIER_PATH,
                       LOG_FILE_NAME);
        
        // Set up controller for detailed algorithm
        RPPGDetailed detailed = RPPGDetailed();
        detailed.load(WIDTH, HEIGHT, TIME_BASE,
                      FACE_CLASSIFIER_PATH,
                      LEFT_EYE_CLASSIFIER_PATH,
                      RIGHT_EYE_CLASSIFIER_PATH,
                      POSE_ESTIMATOR_PATH,
                      LOG_FILE_NAME);
        
        cout << "START ALGORITHM" << endl;
        
        // Run algorithm
        for (int i = 0; i < FRAME_COUNT; i++) {
            
            cout << "======================= FRAME " << i << " =======================" << endl;
            AVFrame * decoded = decoder.GetNextFrame();
            
            if (decoded) {
                
                // OpenCV frame
                cv::Mat frame(decoded->height,
                              decoded->width,
                              CV_8UC3,
                              decoded->data[0]);
                
                // Generate grayframe
                Mat grayFrame;
                cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
                cv::equalizeHist(grayFrame, grayFrame);
                
                double time = decoded->best_effort_timestamp;
                Mat frame1;
                frame.copyTo(frame1);
                Mat frame2;
                frame.copyTo(frame2);
                
                cout << "TIMESTAMP: " << decoded->best_effort_timestamp << endl;
                
                simple.processFrame(frame, grayFrame, time);
                simpleBox.processFrame(frame1, time);
                detailed.processFrame(frame2, time);
                
                imshow("SIMPLE ALGORITHM", frame);
                imshow("SIMPLE BOX ALGORITHM", frame1);
                imshow("DETAILED ALGORITHM", frame2);
                
                if (waitKey(30) >= 0) break;
                
            } else {
                
                cout << "FRAME IS EMPTY!" << endl;
                break;
            }
        }
        av_frame_free(&decoded);
        decoder.CloseFile();
    }
    
    return 0;
}

/*
int main(int argc, const char** argv) {
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }
    
    FFmpegEncoder encoder;
    
    const int WIDTH  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    const double FPS = cap.get(CV_CAP_PROP_FPS);
    const long MSEC = cap.get(CV_CAP_PROP_POS_MSEC);
    cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
    cout << "FPS: " << FPS << endl;
    cout << "MSEC: " << MSEC << endl;

    encoder.OpenFile(argv[2], WIDTH, HEIGHT, 1000*1000, 30, 0.001, 15);
    AVFrame *encode = av_frame_alloc();
    
    Mat frame;
    int64_t begin = (cv::getTickCount()*1000.0)/cv::getTickFrequency();
    
    // Main loop
    while (true) {
        
        cap.read(frame);
        int64_t now = (cv::getTickCount()*1000.0)/cv::getTickFrequency();
        
        if (frame.empty()) {
            while (waitKey() != 27) {}
            break;
        }
        
        // Write frames from simple algorithm to file
        cv::Size frameSize = frame.size();
        avpicture_fill((AVPicture *)encode, frame.data, AV_PIX_FMT_BGR24, frameSize.width, frameSize.height);
        encode->pts = now-begin;
        encode->width = frameSize.width;
        encode->height = frameSize.height;
        encode->format = AV_PIX_FMT_BGR24;
        encoder.WriteFrame(encode, now-begin);
        
        std::cout << "PTS: " << now-begin << std::endl;
        
        imshow("Hay", frame);
        
        if (waitKey(1) == 27) {
            break;
        }
    }
    
    encoder.WriteBufferedFrames();
    encoder.CloseFile();
    av_frame_free(&encode);
    
    return 0;
} */