//
//  RPPG.hpp
//  Heartbeat
//Neel Ganesh 11/26/25


#ifndef RPPG_hpp
#define RPPG_hpp

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include "core/AdaptiveROI.hpp"
#include "core/MultiRegionFusion.hpp"

#include <stdio.h>

using namespace cv;
using namespace dnn;
using namespace std;

enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep };

class RPPG {

public:

    // Constructor
    RPPG() : meanBpm(0.0), minBpm(0.0), maxBpm(0.0), smoothedBpm(0.0) {;}

    // Load Settings
    bool load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
              const int width, const int height, const double timeBase, const int downsample,
              const double samplingFrequency, const double rescanFrequency,
              const int minSignalSize, const int maxSignalSize,
              const string &logPath, const string &haarPath,
              const string &dnnProtoPath, const string &dnnModelPath,
              const bool log, const bool gui);

    void processFrame(Mat &frameRGB, Mat &frameGray, int time);

    void exit();

    typedef vector<Point2f> Contour2f;

private:

    void detectFace(Mat &frameRGB, Mat &frameGray);
    void setNearestBox(vector<Rect> boxes);
    void detectCorners(Mat &frameGray);
    void trackFace(Mat &frameGray);
    void updateMask(Mat &frameGray);
    void updateROI();
    void updateMultiRegionMask(Mat &frameGray);
    void extractSignal_g();
    void extractSignal_pca();
    void extractSignal_xminay();
    void estimateHeartrate();
    void draw(Mat &frameRGB);
    void invalidateFace();
    void log();

    // The algorithm
    rPPGAlgorithm rPPGAlg;

    // The classifier
    faceDetAlgorithm faceDetAlg;
    CascadeClassifier haarClassifier;
    Net dnnClassifier;

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
    Mat1d powerSpectrum;
    double bpm = 0.0;
    double meanBpm;
    double minBpm;
    double maxBpm;
    
    // Signal history for graph display
    vector<double> signalHistory;
    static const int MAX_HISTORY_SIZE = 200;  // Store last 200 samples
    
    // BPM smoothing to prevent random drops
    vector<double> bpmHistory;
    static const int BPM_HISTORY_SIZE = 30;  // Store last 30 BPM readings for better smoothing
    double smoothedBpm;
    
    // Outlier rejection thresholds
    static constexpr double MIN_REASONABLE_BPM = 65.0;  // Floor to keep displayed HR in realistic resting range
    static constexpr double MAX_REASONABLE_BPM = 180.0;  // Maximum reasonable heart rate
    static constexpr double MAX_BPM_DROP_PER_READING = 6.0;   // Allow natural downtrends without sudden crashes
    static constexpr double MAX_BPM_RISE_PER_READING = 10.0;  // Allow natural rises without large spikes
    static constexpr double AVERAGE_VALIDITY_SECONDS = 5.0;   // Invalidate mean/min/max after this timeout
    
    // Multi-region ROI support
    AdaptiveROI adaptiveROI;
    MultiRegionFusion regionFusion;
    vector<ROIRegion> currentRegions;
    Mat1b multiRegionMask;  // Combined mask for all regions

    // Logfiles
    ofstream logfile;
    ofstream logfileDetailed;
    string logfilepath;
};


#endif /* RPPG_hpp */
