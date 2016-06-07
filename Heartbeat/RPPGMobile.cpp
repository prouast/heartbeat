//
//  RPPGMobile.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 21/05/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "RPPGMobile.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include "opencv.hpp"

using namespace cv;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.2
#define MIN_SIGNAL_SIZE 4
#define MAX_SIGNAL_SIZE 10
#define SEC_PER_MIN 60

#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 20

bool RPPGMobile::load(const int width, const int height,
                      const double timeBase,
                      const double samplingFrequency, const double rescanFrequency,
                      const string &logFileName,
                      const string &classifierFilename,
                      const bool log, const bool draw) {
    
    this->minFaceSize = cv::Size(cv::min(width, height) * REL_MIN_FACE_SIZE, cv::min(width, height) * REL_MIN_FACE_SIZE);
    this->rescanFrequency = rescanFrequency;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;
    this->logMode = log;
    this->drawMode = draw;
    this->rescanFlag = false;
    this->lastSamplingTime = 0;

    // Load classifiers
    classifier.load(classifierFilename);
    
    // Setting up logfilepath
    std::ostringstream path_1;
    path_1 << logFileName << "_mobile";
    this->logfilepath = path_1.str();
    
    // Logging bpm according to sampling frequency
    std::ostringstream path_2;
    path_2 << logfilepath << "_bpm.csv";
    logfile.open(path_2.str());
    logfile << "time;face_valid;mean;min;max;mean_ws;min_ws;max_ws\n";
    logfile.flush();
    
    // Logging bpm detailed
    std::ostringstream path_3;
    path_3 << logfilepath << "_bpmAll.csv";
    logfileDetailed.open(path_3.str());
    logfileDetailed << "time;face_valid;bpm;bpm_ws\n";
    logfileDetailed.flush();
    
    return true;
}

void RPPGMobile::exit() {
    logfile.close();
    logfileDetailed.close();
}

void RPPGMobile::processFrame(cv::Mat &frameRGB, cv::Mat &frameGray, int64_t time) {
        
    // Set time
    this->time = time;
    
    if (!faceValid) {
        
        cout << "Not valid, finding a new face" << endl;
        
        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        
    } else if ((time - lastScanTime) * timeBase >= 1/rescanFrequency) {
        
        cout << "Valid, but rescanning face" << endl;
        
        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        rescanFlag = true;
        
    } else {
        
        cout << "Tracking face" << endl;
        
        trackFace(frameGray);
    }
    
    if (faceValid) {
        
        // Update fps
        fps = getFps(t, timeBase);
        
        // Remove old values from raw signal buffer
        while (s.rows > fps * MAX_SIGNAL_SIZE) {
            push(s);
            push(t);
            push(re);
        }
        
        // New values
        Scalar means = mean(frameRGB, mask);
        
        // Add new values to raw signal buffer
        s.push_back(Mat(1, 3, CV_64F, (double[]){means(0), means(1), means(2)}));
        t.push_back<long>(time);
        
        // Save rescan flag
        re.push_back<bool>(rescanFlag);

        // Update fps
        fps = getFps(t, timeBase);
        
        // If valid signal is large enough: estimate
        if (s.rows / fps >= MIN_SIGNAL_SIZE) {
            
            extractSignal_den_detr_mean();
            //extractSignal_xminay();
            
            // PSD estimation
            estimateHeartrate();
        }
        
        if (drawMode) {
            draw(frameRGB);
        }
    }
    
    log();
    
    rescanFlag = false;
    
    frameGray.copyTo(lastFrameGray);
}

void RPPGMobile::detectFace(cv::Mat &frameRGB, cv::Mat &frameGray) {
    
    cout << "Scanning for faces…" << endl;
    
    // Detect faces with Haar classifier
    std::vector<cv::Rect> boxes;
    classifier.detectMultiScale(frameGray, boxes, 1.1, 2, CV_HAAR_SCALE_IMAGE, minFaceSize);
    
    if (boxes.size() > 0) {
        
        cout << "Found a face" << endl;
        
        setNearestBox(boxes);
        detectCorners(frameGray);
        updateROI();
        updateMask(frameGray);
        faceValid = true;
        
    } else {
        
        cout << "Found no face" << endl;
        invalidateFace();
    }
}

void RPPGMobile::setNearestBox(std::vector<cv::Rect> boxes) {
    int index = 0;
    cv::Point p = box.tl() - boxes.at(0).tl();
    int min = p.x * p.x + p.y * p.y;
    for (int i = 1; i < boxes.size(); i++) {
        p = box.tl() - boxes.at(i).tl();
        int d = p.x * p.x + p.y * p.y;
        if (d < min) {
            min = d;
            index = i;
        }
    }
    box = boxes.at(index);
}

void RPPGMobile::detectCorners(cv::Mat &frameGray) {
    
    // Define tracking region
    cv::Mat trackingRegion = cv::Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
    Point points[1][4];
    points[0][0] = Point(box.tl().x + 0.22 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][1] = Point(box.tl().x + 0.78 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][2] = Point(box.tl().x + 0.70 * box.width,
                         box.tl().y + 0.65 * box.height);
    points[0][3] = Point(box.tl().x + 0.30 * box.width,
                         box.tl().y + 0.65 * box.height);
    const Point *pts[1] = {points[0]};
    int npts[] = {4};
    cv::fillPoly(trackingRegion, pts, npts, 1, cv::WHITE);
    
    // Apply corner detection
    goodFeaturesToTrack(frameGray,
                        corners,
                        MAX_CORNERS,
                        QUALITY_LEVEL,
                        MIN_DISTANCE,
                        trackingRegion,
                        3,
                        false,
                        0.04);
}

void RPPGMobile::trackFace(cv::Mat &frameGray) {
    
    // Make sure enough corners are available
    if (corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }
    
    Contour2f corners_1;
    Contour2f corners_0;
    std::vector<uchar> cornersFound_1;
    std::vector<uchar> cornersFound_0;
    cv::Mat err;
    
    // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
    cv::calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);
    // Backtrack once to make it more robust
    cv::calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);
    
    // Exclude no-good corners
    Contour2f corners_1v;
    Contour2f corners_0v;
    for (size_t j = 0; j < corners.size(); j++) {
        if (cornersFound_1[j] && cornersFound_0[j]
            && cv::norm(corners[j]-corners_0[j]) < 2) {
            corners_0v.push_back(corners_0[j]);
            corners_1v.push_back(corners_1[j]);
        } else {
            std::cout << "Mis!" << std::endl;
        }
    }
    
    if (corners_1v.size() >= MIN_CORNERS) {
        
        // Save updated features
        corners = corners_1v;
        
        // Estimate affine transform
        Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);
        
        // Update box
        Contour2f boxCoords;
        boxCoords.push_back(box.tl());
        boxCoords.push_back(box.br());
        Contour2f transformedBoxCoords;
        cv::transform(boxCoords, transformedBoxCoords, transform);
        box = Rect(transformedBoxCoords[0], transformedBoxCoords[1]);
        
        // Update roi
        Contour2f roiCoords;
        roiCoords.push_back(roi.tl());
        roiCoords.push_back(roi.br());
        Contour2f transformedRoiCoords;
        cv::transform(roiCoords, transformedRoiCoords, transform);
        roi = Rect(transformedRoiCoords[0], transformedRoiCoords[1]);
        
        updateMask(frameGray);
        
    } else {
        std::cout << "Tracking failed! Not enough corners left." << std::endl;
        invalidateFace();
    }
}

void RPPGMobile::updateROI() {
    this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
                     Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height));
}

void RPPGMobile::updateMask(cv::Mat &frameGray) {
    
    cout << "Update mask" << endl;
    
    mask = cv::Mat::zeros(frameGray.rows, frameGray.cols, CV_8U);
    rectangle(mask, this->roi, WHITE, FILLED);
}

void RPPGMobile::invalidateFace() {
    
    s = Mat1d();
    t = Mat1d();
    faceValid = false;
}

void RPPGMobile::extractSignal_xminay() {
    
    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_32F);
    denoise(s, re, s_den);
    
    // Normalize raw signals
    Mat s_n = Mat(s_den.rows, s_den.cols, CV_32F);
    normalization(s_den, s_n);
    
    // Calculate X_s signal
    Mat x_s = Mat(s.rows, s.cols, CV_32F);
    addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);
    
    // Calculate Y_s signal
    Mat y_s = Mat(s.rows, s.cols, CV_32F);
    addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
    addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);
    
    const int total = s.rows;
    const int low = (int)(total * LOW_BPM / SEC_PER_MIN / fps);
    const int high = (int)(total * HIGH_BPM / SEC_PER_MIN / fps) + 1;
    
    // Bandpass
    Mat x_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(x_s, x_f, low, high);
    Mat y_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(y_s, y_f, low, high);
    
    // Calculate alpha
    Scalar mean_x_f;
    Scalar stddev_x_f;
    meanStdDev(x_f, mean_x_f, stddev_x_f);
    Scalar mean_y_f;
    Scalar stddev_y_f;
    meanStdDev(y_f, mean_y_f, stddev_y_f);
    double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];
    
    // Calculate signal
    addWeighted(x_f, 1, y_f, -alpha, 0, s_f);
    
    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "r;g;b;r_den;g_den;b_den;x_s;y_s;x_f;y_f;signal\n";
        for (int i = 0; i < s.rows; i++) {
            log << s.at<double>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s.at<double>(i, 2) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_den.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 2) << ";";
            log << x_s.at<double>(i, 0) << ";";
            log << y_s.at<double>(i, 0) << ";";
            log << x_f.at<float>(i, 0) << ";";
            log << y_f.at<float>(i, 0) << ";";
            log << s_f.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPGMobile::extractSignal_den_detr_mean() {
    
    // Denoise
    Mat signalDenoised;
    denoise(s.col(1), re, signalDenoised);
    
    // Normalise
    normalization(signalDenoised, signalDenoised);
    
    // Detrend
    Mat signalDetrended;
    detrend(signalDenoised, signalDetrended, fps);
    
    // Moving average
    Mat signalMeaned;
    movingAverage(signalDetrended, signalMeaned, 3, fps/3);
    signalMeaned.copyTo(s_f);
    
    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "g;g_den;g_detr;g_avg\n";
        for (int i = 0; i < s.rows; i++) {
            log << s.at<double>(i, 1) << ";";
            log << signalDenoised.at<double>(i, 0) << ";";
            log << signalDetrended.at<double>(i, 0) << ";";
            log << signalMeaned.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPGMobile::estimateHeartrate() {
    
    powerSpectrum = cv::Mat(s_f.size(), CV_32F);
    timeToFrequency(s_f, powerSpectrum, true);
    
    // band mask
    const int total = s_f.rows;
    const int low = (int)(total * LOW_BPM / SEC_PER_MIN / fps);
    const int high = (int)(total * HIGH_BPM / SEC_PER_MIN / fps) + 1;
    Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
    bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);
    
    if (!powerSpectrum.empty()) {
        
        // grab index of max power spectrum
        double min, max;
        Point pmin, pmax;
        minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask);
        
        // calculate BPM
        bpm = pmax.y * fps / total * SEC_PER_MIN;
        bpms.push_back(bpm);
        
        // calculate BPM based on weighted squares power spectrum
        double weightedSquares = weightedSquaresMeanIndex(powerSpectrum, low, high);
        bpm_ws = weightedSquares * fps / total * SEC_PER_MIN;
        bpms_ws.push_back(bpm_ws);
        
        cout << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << " BPM=" << bpm << " BPM_WS=" << bpm_ws << endl;
        
        // Logging
        if (logMode) {
            std::ofstream log;
            std::ostringstream filepath;
            filepath << logfilepath << "_estimation_" << time << ".csv";
            log.open(filepath.str());
            log << "i;powerSpectrum\n";
            for (int i = 0; i < powerSpectrum.rows; i++) {
                if (low <= i && i <= high) {
                    log << i << ";";
                    log << powerSpectrum.at<double>(i, 0) << "\n";
                }
            }
            log.close();
        }
    }
    
    if ((time - lastSamplingTime) * timeBase >= 1/samplingFrequency) {
        lastSamplingTime = time;
        
        cv::sort(bpms, bpms, SORT_EVERY_COLUMN);
        cv::sort(bpms_ws, bpms_ws, SORT_EVERY_COLUMN);
        
        // average calculated BPMs since last sampling time
        meanBpm = mean(bpms)(0);
        minBpm = bpms.at<double>(0, 0);
        maxBpm = bpms.at<double>(bpms.rows-1, 0);
        
        meanBpm_ws = mean(bpms_ws)(0);
        minBpm_ws = bpms_ws.at<double>(0, 0);
        maxBpm_ws = bpms_ws.at<double>(bpms_ws.rows-1, 0);
        
        std::cout << "meanBPM=" << meanBpm << " minBpm=" << minBpm << " maxBpm=" << maxBpm << std::endl;

        bpms.pop_back(bpms.rows);
        bpms_ws.pop_back(bpms_ws.rows);
    }
}

void RPPGMobile::log() {
    
    if (lastSamplingTime == time || lastSamplingTime == 0) {
        logfile << time << ";";
        logfile << faceValid << ";";
        logfile << meanBpm << ";";
        logfile << minBpm << ";";
        logfile << maxBpm << ";";
        logfile << meanBpm_ws << ";";
        logfile << minBpm_ws << ";";
        logfile << maxBpm_ws << "\n";
        logfile.flush();
    }
    
    logfileDetailed << time << ";";
    logfileDetailed << faceValid << ";";
    logfileDetailed << bpm << ";";
    logfileDetailed << bpm_ws << "\n";
    logfileDetailed.flush();
}

void RPPGMobile::draw(cv::Mat &frameRGB) {
    
    // Draw roi
    rectangle(frameRGB, roi, cv::GREEN);
    
    // Draw face shape
    ellipse(frameRGB,
            Point(box.tl().x + box.width / 2.0, box.tl().y + box.height / 2.0),
            Size(box.width / 2.5, box.height / 2.0),
            0, 0, 360, cv::GREEN);
    
    // Draw signal
    if (!s_f.empty() && !powerSpectrum.empty()) {
        
        // Display of signals with fixed dimensions
        double displayHeight = box.height/2.0;
        double displayWidth = box.width*0.8;
        
        // Draw signal
        double vmin, vmax;
        Point pmin, pmax;
        minMaxLoc(s_f, &vmin, &vmax, &pmin, &pmax);
        double heightMult = displayHeight/(vmax - vmin);
        double widthMult = displayWidth/(s_f.rows - 1);
        double drawAreaTlX = box.tl().x + box.width;
        double drawAreaTlY = box.tl().y;
        Point p1(drawAreaTlX, drawAreaTlY + (vmax - s_f.at<double>(0, 0))*heightMult);
        Point p2;
        for (int i = 1; i < s_f.rows; i++) {
            p2 = Point(drawAreaTlX + i * widthMult, drawAreaTlY + (vmax - s_f.at<double>(i, 0))*heightMult);
            line(frameRGB, p1, p2, RED, 2);
            p1 = p2;
        }
        
        // Draw powerSpectrum
        const int total = s_f.rows;
        const int low = (int)(total * LOW_BPM / SEC_PER_MIN / fps);
        const int high = (int)(total * HIGH_BPM / SEC_PER_MIN / fps) + 1;
        Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);
        minMaxLoc(powerSpectrum, &vmin, &vmax, &pmin, &pmax, bandMask);
        heightMult = displayHeight/(vmax - vmin);
        widthMult = displayWidth/(high - low);
        drawAreaTlX = box.tl().x + box.width;
        drawAreaTlY = box.tl().y + box.height/2.0;
        p1 = Point(drawAreaTlX, drawAreaTlY + (vmax - powerSpectrum.at<double>(low, 0))*heightMult);
        for (int i = low + 1; i <= high; i++) {
            p2 = Point(drawAreaTlX + (i - low) * widthMult, drawAreaTlY + (vmax - powerSpectrum.at<double>(i, 0)) * heightMult);
            line(frameRGB, p1, p2, RED, 2);
            p1 = p2;
        }
    }
    
    std::stringstream ss;
    
    // Draw BPM text
    if (faceValid) {
        ss.precision(3);
        ss << meanBpm << " bpm";
        putText(frameRGB, ss.str(), Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_PLAIN, 2, cv::RED, 2);
    }
    
    // Draw FPS text
    ss.str("");
    ss << fps << " fps";
    putText(frameRGB, ss.str(), Point(box.tl().x, box.br().y + 40), cv::FONT_HERSHEY_PLAIN, 2, cv::GREEN, 2);
    
    // Draw corners
    /// Draw corners detected
    int r = 4;
    for (int i = 0; i < corners.size(); i++) {
        circle(frameRGB, corners[i], r, cv::WHITE, -1, 8, 0);
    }
    
    // Draw noise warning
    //if (!((mode[0] ? v.at<bool>(v.rows-1, 0) : true) &&
    //    (mode[1] ? v.at<bool>(v.rows-1, 1) : true) &&
    //    (mode[2] ? v.at<bool>(v.rows-1, 2) : true))) {
    //    circle(frameRGB, Point(box.tl().x, box.br().y + 60), 10, cv::RED, -1, 8, 0);
    //}
}
