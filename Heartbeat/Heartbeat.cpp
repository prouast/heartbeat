//
//  Heartbeat.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/06/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "Heartbeat.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "RPPGMobile.hpp"
#include "opencv.hpp"
#include "FFmpegDecoder.hpp"
#include "FFmpegEncoder.hpp"

#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1

using namespace cv;

Heartbeat::Heartbeat(int argc_, char * argv_[], bool switches_on_) {
    
    argc = argc_;
    argv.resize(argc);
    copy(argv_, argv_ + argc, argv.begin());
    switches_on = switches_on_;
    
    // map the switches to the actual
    // arguments if necessary
    if (switches_on) {
        
        vector<string>::iterator it1, it2;
        it1 = argv.begin();
        it2 = it1 + 1;
        
        while (true) {
            
            if (it1 == argv.end()) break;
            if (it2 == argv.end()) break;
            
            if ((*it1)[0] == '-')
                switch_map[*it1] = *(it2);
            
            it1++;
            it2++;
        }
    }
}

string Heartbeat::get_arg(int i) {
    
    if (i >= 0 && i < argc)
        return argv[i];
    
    return "";
}

string Heartbeat::get_arg(string s) {
    
    if (!switches_on) return "";
    
    if (switch_map.find(s) != switch_map.end())
        return switch_map[s];
    
    return "";
}

bool to_bool(string s) {
    bool result;
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    istringstream is(s);
    is >> boolalpha >> result;
    return result;
}

int main(int argc, char * argv[]) {
    
    Heartbeat cmd_line(argc, argv, true);
    
    string input = cmd_line.get_arg("-i"); // Filepath if mode is want offline mode
    
    // Reading rescanIntervl setting
    double rescanFrequency;
    string rescanFrequencyString = cmd_line.get_arg("-r");
    if (rescanFrequencyString != "") {
        rescanFrequency = atof(rescanFrequencyString.c_str());
    } else {
        rescanFrequency = DEFAULT_RESCAN_FREQUENCY;
    }
    
    // Reading samplingFrequency setting
    double samplingFrequency;
    string samplingFrequencyString = cmd_line.get_arg("-f").c_str();
    if (samplingFrequencyString != "") {
        samplingFrequency = atof(samplingFrequencyString.c_str());
    } else {
        samplingFrequency = DEFAULT_SAMPLING_FREQUENCY;
    }
    
    // Reading show setting
    bool show;
    string showString = cmd_line.get_arg("-s");
    if (showString != "") {
        show = to_bool(showString);
    } else {
        show = true;
    }
    
    // Reading log setting
    bool log;
    string logString = cmd_line.get_arg("-d");
    if (logString != "") {
        log = to_bool(logString);
    } else {
        log = false;
    }
    
    const string FACE_CLASSIFIER_PATH = "haarcascade_frontalface_alt.xml";
    
    std::ifstream test(FACE_CLASSIFIER_PATH);
    if (!test) {
        std::cout << "Face classifier xml not found!" << std::endl;
        exit(0);
    }
    
    FFmpegDecoder decoder;
    
    // Working with an input file
    if (input != "") {
        
        if (decoder.OpenFile(input)) {
            
            cout << "Processing input file " << input << endl;
            
            // Strip file extension from input name
            const string LOG_FILE_NAME = input.substr(0, input.find_last_of("."));
            
            // Load video information
            const int WIDTH = decoder.GetWidth();
            const int HEIGHT = decoder.GetHeight();
            const double FPS = decoder.GetFPS();
            const double TIME_BASE = decoder.GetTimeBase();
            
            // Print video information
            cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
            cout << "FPS: " << FPS << endl;
            cout << "TIME BASE: " << TIME_BASE << endl;
            cout << "START TIME: " << decoder.GetStartTime() << endl;
            
            // allocate and init a re-usable frame
            AVFrame *decoded;
            decoded = av_frame_alloc();
            if (!decoded) {
                fprintf(stderr, "Could not allocate video frame\n");
                return false;
            }
            
            RPPGMobile mobile = RPPGMobile();
            mobile.load(WIDTH, HEIGHT, TIME_BASE,
                        samplingFrequency, rescanFrequency,
                        LOG_FILE_NAME, FACE_CLASSIFIER_PATH,
                        log, show);
            
            cout << "START ALGORITHM" << endl;
            
            int i = 0;
            while ((decoded = decoder.GetNextFrame())) {
                
                cout << "===== FRAME " << i << " =====" << endl;
                
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
                    
                    cout << "TIMESTAMP: " << decoded->best_effort_timestamp << endl;
                    
                    mobile.processFrame(frame, grayFrame, time);
                    
                    if (show) {
                        imshow("MOBILE ALGORITHM", frame);
                        if (waitKey(30) >= 0) break;
                    }
                    
                    decoder.FreeBuffer();
                    
                    i++;
                    
                } else {
                    
                    cout << "FRAME IS EMPTY!" << endl;
                    break;
                }
            }
            
            //av_free(decoded);
            av_frame_free(&decoded);
            decoder.CloseFile();
            
        } else {
            fprintf(stderr, "Please provide a valid input file path\n");
            exit(0);
        }
    }
    
    // Working with live feed
    else {
        
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            return -1;
        }
        
        // Configure logfile path
        std::ostringstream filepath;
        filepath << "Live_ffmpeg";
        const string LOG_FILE_NAME = filepath.str();
        
        const int WIDTH  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        const int HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        const double FPS = cap.get(CV_CAP_PROP_FPS);
        const long MSEC = cap.get(CV_CAP_PROP_POS_MSEC);
        const double TIME_BASE = 0.001;
        cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
        cout << "FPS: " << FPS << endl;
        cout << "MSEC: " << MSEC << endl;
        
        RPPGMobile mobile = RPPGMobile();
        mobile.load(WIDTH, HEIGHT, TIME_BASE,
                    samplingFrequency, rescanFrequency,
                    LOG_FILE_NAME, FACE_CLASSIFIER_PATH,
                    log, show);
        
        Mat frame;
        
        // Main loop
        while (true) {
            
            cap.read(frame);
            
            int64_t now = (cv::getTickCount()*1000.0)/cv::getTickFrequency();
            
            // Generate grayframe
            Mat grayFrame;
            cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
            cv::equalizeHist(grayFrame, grayFrame);
            
            if (frame.empty()) {
                while (waitKey() != 27) {}
                break;
            }
            
            mobile.processFrame(frame, grayFrame, now);
            
            if (show) {
                imshow("Live", frame);
            }
            
            if (waitKey(1) == 27) {
                break;
            }
        }
        
    }
    
    return 0;
}

// Old: With video encoding
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