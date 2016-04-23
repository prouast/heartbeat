//
//  RPPGSimple.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 29/02/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPGSimple_hpp
#define RPPGSimple_hpp

#include <string>
#include <stdio.h>
#include <fstream>
#include <opencv2/objdetect/objdetect.hpp>

class RPPGSimple {
    
public:
    
    // Constructor
    RPPGSimple(const int width, const int height,
               const double timeBase,
               const int samplingFrequency, const int rescanInterval,
               const std::string &logFileName,
               const std::string &faceClassifierFilename,
               const std::string &leftEyeClassifierFilename,
               const std::string &rightEyeClassifierFilename,
               const bool log, const bool draw);
    
    void exit();
    void processFrame(cv::Mat &frameRGB, cv::Mat &frameGray, long time);
    
private:
    
    void detectFace(cv::Mat &frameRGB, cv::Mat &frameGray);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectEyes(cv::Mat &frameRGB);
    void updateMask();
    void extractSignal_den_detr_mean();
    void extractSignal_den_band();
    void estimateHeartrate();
    void draw(cv::Mat &frameRGB);
        
    // The classifiers
    cv::CascadeClassifier faceClassifier;
    cv::CascadeClassifier leftEyeClassifier;
    cv::CascadeClassifier rightEyeClassifier;
    
    // Settings
    cv::Size minFaceSize;
    double rescanInterval;
    int samplingFrequency;
    double timeBase;
    bool logMode;
    bool drawMode;
    
    // State variables
    long time;
    double fps;
    double lastSamplingTime;
    double lastScanTime;
    long now;
    bool valid;
    bool updateFlag;
    
    // Mask
    cv::Rect box;
    cv::Rect rightEye;
    cv::Rect leftEye;
    cv::Mat mask;
    
    // Signal
    cv::Mat1d g;
    cv::Mat1d t;
    cv::Mat1d jumps;
    cv::Mat1d signal;
    cv::Mat1d bpms;
    cv::Mat1d powerSpectrum;
    double meanBpm;
    double minBpm;
    double maxBpm;
    
    // Logfiles
    std::ofstream logfile;
    std::ofstream logfileDetailed;
    std::string logfilepath;
};

#endif /* RPPGSimple_hpp */
