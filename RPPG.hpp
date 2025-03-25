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
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

#include <stdio.h>

//---added zScore code
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iterator>
#include <numeric>

typedef long double ld;
typedef unsigned int uint;
typedef std::vector<ld>::iterator vec_iter_ld;

/**
 * Overriding the ostream operator for pretty printing vectors.
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
    os << "[";
    if (vec.size() != 0) {
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(os, " "));
        os << vec.back();
    }
    os << "]";
    return os;
}

/**
 * This class calculates mean and standard deviation of a subvector.
 * This is basically stats computation of a subvector of a window size qual to "lag".
 */

class VectorStats {
public:
    /**
     * Constructor for VectorStats class.
     *
     * @param start - This is the iterator position of the start of the window,
     * @param end   - This is the iterator position of the end of the window,
     */
    VectorStats(vec_iter_ld start, vec_iter_ld end) {
        this->start = start;
        this->end = end;
        this->compute();
    }
    
    /**
     * This method calculates the mean and standard deviation using STL function.
     * This is the Two-Pass implementation of the Mean & Variance calculation.
     */
    void compute() {
        ld sum = std::accumulate(start, end, 0.0);
        uint slice_size = std::distance(start, end);
        ld mean = sum / slice_size;
        std::vector<ld> diff(slice_size);
        std::transform(start, end, diff.begin(), [mean](ld x) { return x - mean; });
        ld sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        ld std_dev = std::sqrt(sq_sum / slice_size);
        
        this->m1 = mean;
        this->m2 = std_dev;
    }
    
    ld mean() {
        return m1;
    }
    
    ld standard_deviation() {
        return m2;
    }
    
private:
    vec_iter_ld start;
    vec_iter_ld end;
    ld m1;
    ld m2;
};

//----original RPPG code
using namespace cv;
using namespace dnn;
using namespace std;

enum rPPGAlgorithm { g, pca, xminay };
enum faceDetAlgorithm { haar, deep };

class RPPG {

public:

    // Constructor
    RPPG() {;}

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

    unordered_map<string, vector<ld>>  z_score_thresholding(vector<ld> input, int lag, ld threshold, ld influence);

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
//    int64_t now;
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

    // Logfiles
    ofstream logfile;
    ofstream logfileDetailed;
    string logfilepath;
};


#endif /* RPPG_hpp */
