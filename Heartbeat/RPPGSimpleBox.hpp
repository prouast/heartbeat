//
//  RPPGSimpleBox.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 6/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPGSimpleBox_hpp
#define RPPGSimpleBox_hpp

#include <string>
#include <stdio.h>
#include <fstream>
#include <opencv2/objdetect/objdetect.hpp>

class RPPGSimpleBox {
    
public:
    RPPGSimpleBox();
    
    void load(const int width, const int height, const double timeBase,
              const std::string &faceClassifierFilename,
              const std::string &leftEyeClassifierFilename,
              const std::string &rightEyeClassifierFilename,
              const std::string &logFilepath);
    void exit();
    void processFrame(cv::Mat &frame, long time);
    
private:
    void detectFace(cv::Mat &frame, cv::Mat &grayFrame);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectEyes(cv::Mat &frame);
    void updateMask();
    void extractSignal_den_detr_mean();
    void extractSignal_den_band();
    void estimateHeartrate();
    void draw(cv::Mat &frame);
    
    cv::CascadeClassifier faceClassifier;
    cv::CascadeClassifier leftEyeClassifier;
    cv::CascadeClassifier rightEyeClassifier;
    
    double rescanInterval;
    int samplingFrequency;
    cv::Size minFaceSize;
    
    long time;
    double timeBase;
    double fps;
    double lastSamplingTime;
    double lastScanTime;
    long now;
    bool valid;
    bool updateFlag;
    
    cv::Rect box;
    cv::Rect rightEye;
    cv::Rect leftEye;
    cv::Mat mask;
    cv::Rect roi;
    
    cv::Mat1d g;
    cv::Mat1d t;
    cv::Mat1d jumps;
    cv::Mat1d signal;
    cv::Mat1d bpms;
    cv::Mat1d powerSpectrum;
    double meanBpm;
    double minBpm;
    double maxBpm;
    std::ofstream logfile;
    std::ofstream logfileDetailed;
    std::string logfilepath;
};

#endif /* RPPGSimpleBox_hpp */
