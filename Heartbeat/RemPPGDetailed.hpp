//
//  RemPPGDetailed.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 3/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RemPPGDetailed_hpp
#define RemPPGDetailed_hpp

#include <string>
#include <stdio.h>
#include <dlib/image_processing.h>
#include <opencv2/objdetect/objdetect.hpp>

class RemPPGDetailed {
    
public:
    RemPPGDetailed();
    
    void load(const int width, const int height, const double timeBase,
              const std::string &faceClassifierFilename,
              const std::string &leftEyeClassifierFilename,
              const std::string &rightEyeClassifierFilename,
              const std::string &poseFilename,
              const std::string &logFilepath);
    void exit();
    void processFrame(cv::Mat &frame, long time);
    
    typedef std::vector<cv::Point> Contour;
    typedef std::vector<cv::Point2f> Contour2f;
    
private:
    void detectFace(cv::Mat &frame, cv::Mat &grayFrame);
    void setNearestBox(std::vector<cv::Rect> boxes);
    void detectFeatures(cv::Mat &frame);
    void detectEyes(cv::Mat &frame);
    void updateMask();
    void trackFace(cv::Mat &grayFrame);
    void extractSignal_den_detr_mean();
    void extractSignal_den_band();
    void estimateHeartrate();
    void draw(cv::Mat &frame);
    
    cv::CascadeClassifier faceClassifier;
    cv::CascadeClassifier leftEyeClassifier;
    cv::CascadeClassifier rightEyeClassifier;
    dlib::shape_predictor pose_model;
    
    double rescanInterval;
    int samplingFrequency;
    cv::Size minFaceSize;
    
    cv::Mat lastGrayFrame;
    
    long time;
    double timeBase;
    double fps;
    double lastSamplingTime;
    double lastScanTime;
    bool rescan;
    double nowTC;
    long now;
    bool valid;
    bool updateFlag;
    
    cv::Rect box;
    cv::Rect rightEye;
    cv::Rect leftEye;
    cv::Mat mask;
    Contour2f contour;
    
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


#endif /* RemPPGDetailed_hpp */
