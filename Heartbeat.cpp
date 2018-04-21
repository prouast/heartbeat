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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv.hpp"
#include "FFmpegDecoder.hpp"
#include "FFmpegEncoder.hpp"
#include "Baseline.hpp"

#define DEFAULT_ALGORITHM "g"
#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 5
#define DEFAULT_MAX_SIGNAL_SIZE 5
#define DEFAULT_DOWNSAMPLE 1 // x means only every xth frame is used

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

rPPGAlgorithm to_algorithm(string s) {
    rPPGAlgorithm result;
    if (s == "g") result = g;
    else if (s == "pca") result = pca;
    else if (s == "xminay") result = xminay;
    else {
        std::cout << "Please specify valid algorithm (g, pca, xminay)!" << std::endl;
        exit(0);
    }
    return result;
}

int main(int argc, char * argv[]) {
    
    Heartbeat cmd_line(argc, argv, true);
    
    string input = cmd_line.get_arg("-i"); // Filepath for offline mode
    
    // algorithm setting
    rPPGAlgorithm algorithm;
    string algorithmString = cmd_line.get_arg("-a");
    if (algorithmString != "") {
        algorithm = to_algorithm(algorithmString);
    } else {
        algorithm = to_algorithm(DEFAULT_ALGORITHM);
    }
    
    cout << "Using algorithm " << algorithm << "." << endl;
    
    // rescanFrequency setting
    double rescanFrequency;
    string rescanFrequencyString = cmd_line.get_arg("-r");
    if (rescanFrequencyString != "") {
        rescanFrequency = atof(rescanFrequencyString.c_str());
    } else {
        rescanFrequency = DEFAULT_RESCAN_FREQUENCY;
    }
    
    // samplingFrequency setting
    double samplingFrequency;
    string samplingFrequencyString = cmd_line.get_arg("-f").c_str();
    if (samplingFrequencyString != "") {
        samplingFrequency = atof(samplingFrequencyString.c_str());
    } else {
        samplingFrequency = DEFAULT_SAMPLING_FREQUENCY;
    }
    
    // max signal size setting
    int maxSignalSize;
    string maxSignalSizeString = cmd_line.get_arg("-max");
    if (maxSignalSizeString != "") {
        maxSignalSize = atof(maxSignalSizeString.c_str());
    } else {
        maxSignalSize = DEFAULT_MAX_SIGNAL_SIZE;
    }
    
    // min signal size setting
    int minSignalSize;
    string minSignalSizeString = cmd_line.get_arg("-min");
    if (minSignalSizeString != "") {
        minSignalSize = atof(minSignalSizeString.c_str());
    } else {
        minSignalSize = DEFAULT_MIN_SIGNAL_SIZE;
    }
    
    // visualize baseline setting
    string baseline_input = cmd_line.get_arg("-baseline");
    
    if (minSignalSize > maxSignalSize) {
        std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
        exit(0);
    }
    
    // Reading gui setting
    bool gui;
    string guiString = cmd_line.get_arg("-gui");
    if (guiString != "") {
        gui = to_bool(guiString);
    } else {
        gui = true;
    }
    
    // Reading log setting
    bool log;
    string logString = cmd_line.get_arg("-log");
    if (logString != "") {
        log = to_bool(logString);
    } else {
        log = false;
    }
    
    // Reading downsample setting
    int downsample;
    string downsampleString = cmd_line.get_arg("-ds");
    if (downsampleString != "") {
        downsample = atof(downsampleString.c_str());
    } else {
        downsample = DEFAULT_DOWNSAMPLE;
    }
    
    const string CLASSIFIER_PATH = "haarcascade_frontalface_alt.xml";
    
    std::ifstream test(CLASSIFIER_PATH);
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
            const string LOG_PATH = input.substr(0, input.find_last_of("."));
            
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
            
            std::ostringstream title;
            title << "rPPG offline - " << WIDTH << "x" << HEIGHT << " -a " << algorithm << " -r " << rescanFrequency << " -f " << samplingFrequency << " -min " << minSignalSize << " -max " << maxSignalSize << " -ds " << downsample;
            
            // allocate and init a re-usable frame
            AVFrame *decoded;
            decoded = av_frame_alloc();
            if (!decoded) {
                fprintf(stderr, "Could not allocate video frame\n");
                return false;
            }
            
            RPPG rppg = RPPG();
            rppg.load(algorithm,
                      WIDTH, HEIGHT, TIME_BASE, downsample,
                      samplingFrequency, rescanFrequency,
                      minSignalSize, maxSignalSize,
                      LOG_PATH, CLASSIFIER_PATH,
                      log, gui);
            
            Baseline baseline = Baseline();
            if (baseline_input != "") {
                baseline.load(1, 0.000001, baseline_input);
            }
            
            cout << "START ALGORITHM" << endl;
            
            int i = 0;
            while ((decoded = decoder.GetNextFrame())) {
                
                cout << "===== FRAME " << i << " =====" << endl;
                
                if (decoded) {
                    
                    // OpenCV frame
                    cv::Mat frameRGB(decoded->height,
                                     decoded->width,
                                     CV_8UC3,
                                     decoded->data[0]);
                    
                    // Generate grayframe
                    Mat frameGray;
                    cv::cvtColor(frameRGB, frameGray, CV_BGR2GRAY);
                    cv::equalizeHist(frameGray, frameGray);
                    
                    double time = decoded->best_effort_timestamp;
                    
                    cout << "TIMESTAMP: " << time << endl;
                    
                    if (i % downsample == 0) {
                        
                        rppg.processFrame(frameRGB, frameGray, time);
                    
                    } else {
                        
                        cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
                    }
                    
                    if (baseline_input != "") {
                        baseline.processFrame(frameRGB, time);
                    }
                    
                    if (gui) {
                        imshow(title.str(), frameRGB);
                        if (waitKey(30) >= 0) break;
                    }
                    
                    frameRGB.deallocate();
                    decoder.FreeBuffer();
                    
                    i++;
                    
                } else {
                    
                    cout << "FRAME IS EMPTY!" << endl;
                    break;
                }
            }
            
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
        const string LOG_PATH = filepath.str();
        
        const int WIDTH  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        const int HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        const double FPS = cap.get(CV_CAP_PROP_FPS);
        const long MSEC = cap.get(CV_CAP_PROP_POS_MSEC);
        const double TIME_BASE = 0.001;
        cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
        cout << "FPS: " << FPS << endl;
        cout << "MSEC: " << MSEC << endl;
        
        std::ostringstream title;
        title << "rPPG online - " << WIDTH << "x" << HEIGHT << " -a " << algorithm << " -r " << rescanFrequency << " -f " << samplingFrequency << " -min " << minSignalSize << " -max " << maxSignalSize << " -ds " << downsample;
        
        RPPG rppg = RPPG();
        rppg.load(algorithm,
                  WIDTH, HEIGHT, TIME_BASE, downsample,
                  samplingFrequency, rescanFrequency,
                  minSignalSize, maxSignalSize,
                  LOG_PATH, CLASSIFIER_PATH,
                  log, gui);
        
        Mat frameRGB;
        
        int i = 0;
        
        // Main loop
        while (true) {
            
            cap.read(frameRGB);
            
            if (i % downsample == 0) {
                
                int64_t time = (cv::getTickCount()*1000.0)/cv::getTickFrequency();
                
                // Generate grayframe
                Mat frameGray;
                cv::cvtColor(frameRGB, frameGray, CV_BGR2GRAY);
                cv::equalizeHist(frameGray, frameGray);
                
                if (frameRGB.empty()) {
                    while (waitKey() != 27) {}
                    break;
                }
                
                rppg.processFrame(frameRGB, frameGray, time);
                
                if (gui) {
                    imshow(title.str(), frameRGB);
                }
            
            } else {
                
                cout << "SKIPPING FRAME TO DOWNSAMPLE!" << endl;
            }
            
            if (waitKey(30) == 27) {
                break;
            }
            
            i++;
        }
        
    }
    
    return 0;
}
