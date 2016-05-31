//
//  RPPGMobile.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 21/05/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPGMobile_hpp
#define RPPGMobile_hpp

#include <string>
#include <stdio.h>
#include <fstream>
#include <opencv2/objdetect/objdetect.hpp>

class RPPGMobile {
    
public:
    
    // Constructor
    RPPGMobile() {;}
    
    // Load Settings
    bool load(const int width, const int height,
              const double timeBase,
              const int samplingFrequency, const int rescanInterval,
              const std::string &logFileName,
              const std::string &classifierFilename,
              const bool log, const bool draw);
    
    void processFrame(cv::Mat &frameRGB, cv::Mat &frameGray, int64_t time);
    
    void exit();
    
    typedef std::vector<cv::Point2f> Contour2f;
    
private:
    
    void detectFace(cv::Mat &frameRGB, cv::Mat &frameGray);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectCorners(cv::Mat &frameGray);
    void trackFace(cv::Mat &frameGray);
    void updateMask(cv::Mat &frameGray);
    void extractSignal();
    void extractSignal_den_detr_mean();
    void estimateHeartrate();
    void draw(cv::Mat &frameRGB);
    
    // The classifiers
    cv::CascadeClassifier classifier;
    
    // Settings
    cv::Size minFaceSize;
    double rescanInterval;
    int samplingFrequency;
    double timeBase;
    bool logMode;
    bool drawMode;
    
    // State variables
    int64_t time;
    double fps;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int64_t now;
    bool faceValid;
    bool rescanFlag;
    bool mode[3] = {false, true, false};
    
    // Tracking
    cv::Mat lastFrameGray;
    Contour2f corners;
    
    // Mask
    cv::Rect box;
    cv::Mat mask;
    cv::Rect roi;
    
    // Raw signal
    cv::Mat1d s;
    cv::Mat1d t;
    cv::Mat1b re;
    
    // Signal validation
    cv::Mat1b v;
    bool s_flags[3] = {false, false, false};
    
    // Valid signal and estimation
    cv::Mat1d s_v;
    cv::Mat1d s_f;
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

#endif /* RPPGMobile_hpp */
