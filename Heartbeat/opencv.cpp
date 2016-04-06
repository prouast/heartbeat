//
//  opencv.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 3/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "opencv.hpp"
#include <limits>

#include "opencv2/highgui/highgui.hpp"
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
            //printMat<long>("t", t);
            //std::cout << "diff = " << t.at<long>(t.rows-1, 0) << " - " << t.at<long>(0, 0) << " * " << timeBase << std::endl;
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
    
    void denoiseFilter2(InputArray _a, OutputArray _b, Mat &jumps) {
        
        Mat a = _a.getMat();
        
        Mat diff;
        subtract(a.rowRange(1, a.rows), a.rowRange(0, a.rows-1), diff);
        
        for (int i = 0; i < jumps.rows; i++) {
            if (jumps.at<bool>(i, 0)) {
                Mat mask = Mat::zeros(a.col(0).size(), CV_8U);
                mask.rowRange(i, mask.rows).setTo(ONE);
                add(a, Scalar(-diff.at<double>(i-1, 0)), a, mask);
            }
        }
        
        a.copyTo(_b);
    }
    
    void detrendFilter(InputArray _a, OutputArray _b, int lambda) {
        
        Mat a = _a.total() == (size_t)_a.size().height ? _a.getMat() : _a.getMat().t();
        if (a.total() < 3) {
            a.copyTo(_b);
        } else {
            int t = (int)a.total();
            Mat i = Mat::eye(t, t, a.type());
            Mat d = Mat(Matx<double,1,3>(1, -2, 1));
            Mat d2Aux = Mat::ones(t-2, 1, a.type()) * d;
            Mat d2 = Mat::zeros(t-2, t, a.type());
            for (int k = 0; k < 3; k++) {
                d2Aux.col(k).copyTo(d2.diag(k));
            }
            Mat b = (i - (i + lambda * lambda * d2.t() * d2).inv()) * a;
            b.copyTo(_b);
        }
    }
    
    void bandpassFilter(cv::InputArray _a, cv::OutputArray _b, double low, double high) {
        
        Mat a = _a.getMat();
        
        if (a.total() < 3) {
            a.copyTo(_b);
        } else {
            // Convert to frequency domain
            Mat frequencySpectrum;
            timeToFrequency(a, frequencySpectrum, false);
            
            // Make the filter
            Mat filter = frequencySpectrum.clone();
            butterworth_bandpass_filter(filter, low, high, 8);
            
            // Apply the filter
            mulSpectrums(frequencySpectrum, filter, frequencySpectrum, 0);
            
            // Convert to time domain
            frequencyToTime(frequencySpectrum, _b);
        }
    }
    
    void meanFilter(InputArray _a, OutputArray _b, int n, int s) {
        _a.getMat().copyTo(_b);
        Mat b = _b.getMat();
        for (size_t i = 0; i < n; i++) {
            cv::blur(b, b, Size(s, s));
        }
    }
    
    void butterworth_lowpass_filter(Mat &filter, double cutoff, int n) {
        CV_DbgAssert(cutoff > 0 && n > 0 && filter.rows % 2 == 0 && filter.cols % 2 == 0);
        
        Mat tmp = Mat(filter.rows, filter.cols, CV_32F);
        double radius;
        
        for (int i = 0; i < filter.rows; i++) {
            for (int j = 0; j < filter.cols; j++) {
                radius = i;
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
        Mat output;
        normalize(outputPlanes[0], output, 0, 1, CV_MINMAX);
        output.copyTo(_b);
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