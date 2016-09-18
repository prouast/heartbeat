//
//  opencv.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 3/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "opencv.hpp"
#include <limits>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

namespace cv {
    
    /* COMMON FUNCTIONS */
    
    double getFps(Mat &t, const double timeBase) {
        
        double result;
        
        if (t.empty()) {
            result = 1.0;
        } else if (t.rows == 1) {
            result = std::numeric_limits<double>::max();
        } else {
            double diff = (t.at<long>(t.rows-1, 0) - t.at<long>(0, 0)) * timeBase;
            result = diff == 0 ? std::numeric_limits<double>::max() : t.rows/diff;
        }
        
        return result;
    }
    
    void push(Mat &m) {
        const int length = m.rows;
        m.rowRange(1, length).copyTo(m.rowRange(0, length - 1));
        m.pop_back();
    }
    
    void plot(cv::Mat &mat) {
        while (true) {
            cv::imshow("plot", mat);
            if (waitKey(30) >= 0) break;
        }
    }
    
    double weightedMeanIndex(InputArray _a, int low, int high) {
        
        double result;
        
        // Create input mats
        Mat a = _a.getMat();
        Mat m = Mat::zeros(a.size(), CV_8U);
        m.rowRange(min(low, a.rows), min(high, a.rows) + 1).setTo(ONE);
        
        CV_Assert(a.type() == CV_64F);
        
        // Normalize the input
        normalize(a, a, 1, 0, NORM_L1, -1, m);
        
        for (int i = low; i <= high; i++) {
            result += a.at<double>(0, i) * i;
        }
        
        return result;
    }
    
    double weightedSquaresMeanIndex(InputArray _a, int low, int high) {
        
        double result;
        
        // Create input mats
        Mat a = _a.getMat().clone();
        Mat m = Mat::zeros(a.size(), CV_8U);
        m.rowRange(min(low, a.rows), min(high, a.rows) + 1).setTo(ONE);
        
        CV_Assert(a.type() == CV_64F);
        
        // Normalize the input range
        normalize(a, a, 1, 0, NORM_L1, -1, m);
        
        //printMat<double>("a", a);
        
        // Quadruple the input array
        multiply(a, a, a);
        multiply(a, a, a);
        
        //printMat<double>("a", a);
        
        // Normalize the adjusted input range
        normalize(a, a, 1, 0, NORM_L1, -1, m);
        
        //printMat<double>("a", a);
        
        for (int i = low; i <= high; i++) {
            result += a.at<double>(0, i) * i;
        }
        
        return result;
    }
    
    /* FILTERS */
    
    // Subtract mean and divide by standard deviation
    void normalization(InputArray _a, OutputArray _b) {
        _a.getMat().copyTo(_b);
        Mat b = _b.getMat();
        Scalar mean, stdDev;
        for (int i = 0; i < b.cols; i++) {
            meanStdDev(b.col(i), mean, stdDev);
            b.col(i) = (b.col(i) - mean[0]) / stdDev[0];
        }
    }
    
    // Eliminate jumps
    void denoise(InputArray _a, InputArray _jumps, OutputArray _b) {
        
        Mat a = _a.getMat().clone();
        Mat jumps = _jumps.getMat().clone();
        
        CV_Assert(a.type() == CV_64F && jumps.type() == CV_8U);
        
        if (jumps.rows != a.rows) {
            jumps.rowRange(jumps.rows-a.rows, jumps.rows).copyTo(jumps);
        }
        
        Mat diff;
        subtract(a.rowRange(1, a.rows), a.rowRange(0, a.rows-1), diff);
        
        for (int i = 1; i < jumps.rows; i++) {
            if (jumps.at<bool>(i, 0)) {
                Mat mask = Mat::zeros(a.size(), CV_8U);
                mask.rowRange(i, mask.rows).setTo(ONE);
                for (int j = 0; j < a.cols; j++) {
                    add(a.col(j), -diff.at<double>(i-1, j), a.col(j), mask.col(j));
                }
            }
        }
        
        a.copyTo(_b);
    }
    
    // Advanced detrending filter based on smoothness priors approach (High pass equivalent)
    void detrend(InputArray _a, OutputArray _b, int lambda) {
        
        Mat a = _a.getMat();
        CV_Assert(a.type() == CV_64F);
        
        // Number of rows
        int rows = a.rows;
        
        if (rows < 3) {
            a.copyTo(_b);
        } else {
            // Construct I
            Mat i = Mat::eye(rows, rows, a.type());
            // Construct D2
            Mat d = Mat(Matx<double,1,3>(1, -2, 1));
            Mat d2Aux = Mat::ones(rows-2, 1, a.type()) * d;
            Mat d2 = Mat::zeros(rows-2, rows, a.type());
            for (int k = 0; k < 3; k++) {
                d2Aux.col(k).copyTo(d2.diag(k));
            }
            // Calculate b = (I - (I + λ^2 * D2^t*D2)^-1) * a
            Mat b = (i - (i + lambda * lambda * d2.t() * d2).inv()) * a;
            b.copyTo(_b);
        }
    }
    
    // Moving average filter (low pass equivalent)
    void movingAverage(InputArray _a, OutputArray _b, int n, int s) {
        
        CV_Assert(s > 0);
        
        _a.getMat().copyTo(_b);
        Mat b = _b.getMat();
        for (size_t i = 0; i < n; i++) {
            cv::blur(b, b, Size(s, s));
        }
    }
    
    // Bandpass filter
    void bandpass(cv::InputArray _a, cv::OutputArray _b, double low, double high) {
        
        Mat a = _a.getMat();
        
        if (a.total() < 3) {
            a.copyTo(_b);
        } else {
            
            // Convert to frequency domain
            Mat frequencySpectrum = Mat(a.rows, a.cols, CV_32F);
            timeToFrequency(a, frequencySpectrum, false);
            
            // Make the filter
            Mat filter = frequencySpectrum.clone();
            butterworth_bandpass_filter(filter, low, high, 8);
            
            // Apply the filter
            multiply(frequencySpectrum, filter, frequencySpectrum);
            
            // Convert to time domain
            frequencyToTime(frequencySpectrum, _b);
        }
    }
    
    void butterworth_lowpass_filter(Mat &filter, double cutoff, int n) {
        CV_DbgAssert(cutoff > 0 && n > 0 && filter.rows % 2 == 0 && filter.cols % 2 == 0);
        
        Mat tmp = Mat(filter.rows, filter.cols, CV_32F);
        //Point centre = Point(filter.rows / 2, filter.cols / 2);
        double radius;
        
        for (int i = 0; i < filter.rows; i++) {
            for (int j = 0; j < filter.cols; j++) {
                radius = i;
                //radius = (double)sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
                tmp.at<float>(i, j) = (float)(1 / (1 + pow(radius / cutoff, 2 * n)));
            }
        }
        
        Mat toMerge[] = {tmp, tmp};
        merge(toMerge, 2, filter);
    }
    
    void butterworth_bandpass_filter(Mat &filter, double cutin, double cutoff, int n) {
        CV_DbgAssert(cutoff > 0 && cutin < cutoff && n > 0 &&
                     filter.rows % 2 == 0 && filter.cols % 2 == 0);
        Mat off = filter.clone();
        butterworth_lowpass_filter(off, cutoff, n);
        Mat in = filter.clone();
        butterworth_lowpass_filter(in, cutin, n);
        filter = off - in;
    }
    
    void timeToFrequency(InputArray _a, OutputArray _b, bool magnitude) {
        
        // Prepare planes
        Mat a = _a.getMat();
        Mat planes[] = {cv::Mat_<float>(a), cv::Mat::zeros(a.size(), CV_32F)};
        Mat powerSpectrum;
        merge(planes, 2, powerSpectrum);
        
        // Fourier transform
        dft(powerSpectrum, powerSpectrum, DFT_COMPLEX_OUTPUT);
        
        if (magnitude) {
            split(powerSpectrum, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            planes[0].copyTo(_b);
        } else {
            powerSpectrum.copyTo(_b);
        }
    }
    
    void frequencyToTime(InputArray _a, OutputArray _b) {
        
        Mat a = _a.getMat();
        
        // Inverse fourier transform
        idft(a, a);
        
        // Split into planes; plane 0 is output
        Mat outputPlanes[2];
        split(a, outputPlanes);
        Mat output = Mat(a.rows, 1, a.type());
        normalize(outputPlanes[0], output, 0, 1, CV_MINMAX);
        output.copyTo(_b);
    }
    
    void pcaComponent(cv::InputArray _a, cv::OutputArray _b, cv::OutputArray _pc, int low, int high) {
        
        Mat a = _a.getMat();
        CV_Assert(a.type() == CV_64F);
        
        // Perform PCA
        cv::PCA pca(a, cv::Mat(), CV_PCA_DATA_AS_ROW);
        
        // Calculate PCA components
        cv::Mat pc = a * pca.eigenvectors.t();
        
        // Band mask
        const int total = a.rows;
        Mat bandMask = Mat::zeros(a.rows, 1, CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);
        
        // Identify most distinct
        std::vector<double> vals;
        for (int i = 0; i < pc.cols; i++) {
            cv::Mat magnitude = Mat(pc.rows, 1, CV_32F);
            // Calculate spectral magnitudes
            cv::timeToFrequency(pc.col(i), magnitude, true);
            // Normalize
            //printMat<float>("magnitude1", magnitude);
            cv::normalize(magnitude, magnitude, 1, 0, NORM_L1, -1, bandMask);
            //printMat<float>("magnitude2", magnitude);
            // Grab index of max
            double min, max;
            Point pmin, pmax;
            cv::minMaxLoc(magnitude, &min, &max, &pmin, &pmax, bandMask);
            vals.push_back(max);
            
        }
        
        // Select most distinct
        int idx[2];
        cv::minMaxIdx(vals, 0, 0, 0, &idx[0]);
        if (idx[0] == -1) {
            pc.col(1).copyTo(_b);
        } else {
            //pc.col(1).copyTo(_b);
            pc.col(idx[1]).copyTo(_b);
        }
        
        pc.copyTo(_pc);
    }
    
    /* LOGGING */
    
    void printMagnitude(String title, Mat &powerSpectrum) {
        Mat planes[2];
        split(powerSpectrum, planes);
        magnitude(planes[0], planes[1], planes[0]);
        Mat mag = (planes[0]).clone();
        mag += Scalar::all(1);
        log(mag, mag);
        printMat<double>(title, mag);
    }
    
    void printMatInfo(const std::string &name, InputArray _a) {
        Mat a = _a.getMat();
        std::cout << name << ": " << a.rows << "x" << a.cols
        << " channels=" << a.channels()
        << " depth=" << a.depth()
        << " isContinuous=" << (a.isContinuous() ? "true" : "false")
        << " isSubmatrix=" << (a.isSubmatrix() ? "true" : "false") << std::endl;
    }
}
