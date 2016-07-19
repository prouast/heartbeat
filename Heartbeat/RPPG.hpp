//
//  RPPG.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>

#include <stdio.h>

using namespace cv;
using namespace std;

enum rPPGAlgorithm { g, pca, xminay };

class RPPG {
    
public:
    
    // Constructor
    RPPG() {;}
    
    // Load Settings
    bool load(const rPPGAlgorithm algorithm,
              const int width, const int height, const double timeBase, const int downsample,
              const double samplingFrequency, const double rescanFrequency,
              const int minSignalSize, const int maxSignalSize,
              const string &logPath, const string &classifierPath,
              const bool log, const bool gui);
    
    void processFrame(Mat &frameRGB, Mat &frameGray, int64_t time);
    
    void exit();
    
    typedef vector<Point2f> Contour2f;
    
private:
    
    void detectFace(Mat &frameRGB, Mat &frameGray);
    void setNearestBox(vector<Rect> boxes);
    void detectCorners(Mat &frameGray);
    void trackFace(Mat &frameGray);
    void updateMask(Mat &frameGray);
    void updateROI();
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    void estimateHeartrate();
    void draw(Mat &frameRGB);
    void invalidateFace();
    void log();
    
    // The algorithm
    rPPGAlgorithm algorithm;
    
    // The classifiers
    CascadeClassifier classifier;
    
    // Settings
    Size minFaceSize;
    int maxSignalSize;
    int minSignalSize;
    double rescanFrequency;
    double samplingFrequency;
    double timeBase;
    bool logMode;
    bool guiMode;
    
    // State variables
    int64_t time;
    double fps;
    int high;
    int64_t lastSamplingTime;
    int64_t lastScanTime;
    int low;
    int64_t now;
    bool faceValid;
    bool signalValid;
    bool rescanFlag;
    
    // Tracking
    Mat lastFrameGray;
    Contour2f corners;
    
    // Mask
    Rect box;
    Mat1b mask;
    Rect roi;
    
    // Raw signal
    Mat1d s;
    Mat1d t;
    Mat1b re;
    
    // Estimation
    Mat1d s_f;
    Mat1d bpms;
    Mat1d bpms_ws;
    Mat1d powerSpectrum;
    double bpm = 0.0;
    double bpm_ws = 0.0;
    double meanBpm;
    double minBpm;
    double maxBpm;
    double meanBpm_ws;
    double minBpm_ws;
    double maxBpm_ws;
    
    // Logfiles
    ofstream logfile;
    ofstream logfileDetailed;
    string logfilepath;
};


#endif /* RPPG_hpp */
