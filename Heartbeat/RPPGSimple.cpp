//
//  RPPGSimple.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 29/02/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "RPPGSimple.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "opencv.hpp"

using namespace cv;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.2
#define SIGNAL_SIZE 10
#define SEC_PER_MIN 60

bool RPPGSimple::load(const int width, const int height,
                      const double timeBase,
                      const int samplingFrequency, const int rescanInterval,
                      const string &logFileName,
                      const string &faceClassifierFilename,
                      const string &leftEyeClassifierFilename,
                      const string &rightEyeClassifierFilename,
                      const bool log, const bool draw) {
    
    this->minFaceSize = cv::Size(cv::min(width, height) * REL_MIN_FACE_SIZE, cv::min(width, height) * REL_MIN_FACE_SIZE);
    this->rescanInterval = rescanInterval;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;
    this->logMode = log;
    this->drawMode = draw;
    this->updateFlag = false;
    this->mask = cv::Mat::zeros(height, width, CV_8UC1);
    
    // Load classifiers
    faceClassifier.load(faceClassifierFilename);
    leftEyeClassifier.load(rightEyeClassifierFilename);
    rightEyeClassifier.load(leftEyeClassifierFilename);
    
    // Setting up logfilepath
    std::ostringstream path_1;
    path_1 << logFileName << "_simple";
    this->logfilepath = path_1.str();
    
    // Logging bpm according to sampling frequency
    std::ostringstream path_2;
    path_2 << logfilepath << "_bpm.csv";
    logfile.open(path_2.str());
    logfile << "time;mean;min;max\n";
    
    // Logging bpm detailed
    std::ostringstream path_3;
    path_3 << logfilepath << "_bpmDetailed.csv";
    logfileDetailed.open(path_3.str());
    logfileDetailed << "time;bpm\n";
    
    return true;
}

void RPPGSimple::exit() {
    logfile.close();
    logfileDetailed.close();
}

void RPPGSimple::processFrame(cv::Mat &frameRGB, cv::Mat &frameGray, long time) {
    
    cout << "================= SIMPLE =================" << endl;
    
    // Set time
    this->time = time;
    
    if (!valid) {
        
        cout << "Not valid, finding a new face" << endl;
        
        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        
    } else if ((time - lastScanTime) * timeBase >= rescanInterval) {
        
        cout << "Valid, but rescanning face" << endl;
        
        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        updateFlag = true;
    }
    
    if (valid) {
        
        fps = getFps(t, timeBase);
        
        // Remove old values from buffer
        while (g.rows > fps * SIGNAL_SIZE) {
            push(g);
            push(t);
            push(jumps);
        }
        
        // Add new values to buffer
        Scalar means = mean(frameRGB, mask);
        g.push_back<double>(means(1));
        jumps.push_back<bool>(updateFlag ? true : false);
        t.push_back<long>(time);
        
        fps = getFps(t, timeBase);
        
        updateFlag = false;
                
        // If buffer is large enough, send off to estimation
        if (g.rows / fps >= SIGNAL_SIZE) {
            
            // Save raw signal
            g.copyTo(signal);
            
            // Apply filters
            extractSignal_den_detr_mean();
            
            // PSD estimation
            estimateHeartrate();
        }
        
        draw(frameRGB);
    }
}

void RPPGSimple::detectFace(cv::Mat &frameRGB, cv::Mat &frameGray) {
    
    cout << "Scanning for faces…" << endl;
    
    // Detect faces with Haar classifier
    std::vector<cv::Rect> boxes;
    faceClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CV_HAAR_SCALE_IMAGE, minFaceSize);
    
    if (boxes.size() > 0) {
        
        cout << "Found a face" << endl;
        
        setNearestBox(boxes);
        detectEyes(frameRGB);
        updateMask();
        valid = true;
        
    } else {
        
        cout << "Found no face" << endl;
        
        valid = false;
    }
}

void RPPGSimple::setNearestBox(std::vector<cv::Rect> boxes) {
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

void RPPGSimple::detectEyes(cv::Mat &frameRGB) {
    
    Rect leftEyeROI = Rect(box.tl().x + box.width/16,
                           box.tl().y + box.height/4.5,
                           (box.width - 2*box.width/16)/2,
                           box.height/3.0);

    
    Rect rightEyeROI = Rect(box.tl().x + box.width/16 + (box.width - 2*box.width/16)/2,
                           box.tl().y + box.height/4.5,
                           (box.width - 2*box.width/16)/2,
                           box.height/3.0);
    
    Mat leftSub = frameRGB(leftEyeROI);
    Mat rightSub = frameRGB(rightEyeROI);
    
    // Detect eyes with Haar classifier
    std::vector<cv::Rect> eyeBoxesLeft;
    leftEyeClassifier.detectMultiScale(leftSub, eyeBoxesLeft, 1.1, 2, 0);
    std::vector<cv::Rect> eyeBoxesRight;
    rightEyeClassifier.detectMultiScale(rightSub, eyeBoxesRight, 1.1, 2, 0);
    
    if (eyeBoxesLeft.size() > 0) {
        Rect leftEye = eyeBoxesLeft.at(0);
        Point tl = Point(leftEyeROI.x + leftEye.x, leftEyeROI.y + leftEye.y);
        Point br = Point(leftEyeROI.x + leftEye.x + leftEye.width,
                         leftEyeROI.y + leftEye.y + leftEye.height);
        this->leftEye = Rect(tl, br);
    } else {
        cout << "No left eye found" << endl;
    }
    
    if (eyeBoxesRight.size() > 0) {
        Rect rightEye = eyeBoxesRight.at(0);
        Point tl = Point(rightEyeROI.x + rightEye.x, rightEyeROI.y + rightEye.y);
        Point br = Point(rightEyeROI.x + rightEye.x + rightEye.width,
                         rightEyeROI.y + rightEye.y + rightEye.height);
        this->rightEye = Rect(tl, br);
    } else {
        cout << "No right eye found" << endl;
    }
}

void RPPGSimple::updateMask() {
    
    cout << "Update mask" << endl;
    
    mask.setTo(BLACK);
    ellipse(mask,
            Point(box.tl().x + box.width/2.0, box.tl().y + box.height/2.0),
            Size(box.width/2.5, box.height/2.0),
            0, 0, 360, WHITE, FILLED);
    circle(mask,
           Point(leftEye.tl().x + leftEye.width/2.0, leftEye.tl().y + leftEye.height/2.0),
           (leftEye.width + leftEye.height)/4.0, BLACK, FILLED);
    circle(mask,
           Point(rightEye.tl().x + rightEye.width/2.0, rightEye.tl().y + rightEye.height/2.0),
           (rightEye.width + rightEye.height)/4.0, BLACK, FILLED);
}

void RPPGSimple::extractSignal_den_detr_mean() {

    // Denoise
    Mat signalDenoised;
    denoise(signal, signalDenoised, jumps);
    
    // Detrend
    Mat signalDetrended;
    detrend(signalDenoised, signalDetrended, fps);
    
    // Moving average
    Mat signalMeaned;
    movingAverage(signalDetrended, signalMeaned, 3, fps/3);
    signalMeaned.copyTo(signal);
    
    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "g;g_den;g_detr;g_avg\n";
        for (int i = 0; i < g.rows; i++) {
            log << g.at<double>(i, 0) << ";";
            log << signalDenoised.at<double>(i, 0) << ";";
            log << signalDetrended.at<double>(i, 0) << ";";
            log << signalMeaned.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPGSimple::estimateHeartrate() {
    
    powerSpectrum = cv::Mat(signal.size(), CV_32F);
    timeToFrequency(signal, powerSpectrum, true);
    
    // band mask
    const int total = signal.rows;
    const int low = (int)(total * LOW_BPM / SEC_PER_MIN / fps);
    const int high = (int)(total * HIGH_BPM / SEC_PER_MIN / fps);
    Mat bandMask = Mat::zeros(signal.size(), CV_8U);
    bandMask.rowRange(min(low, total), min(high, total)).setTo(ONE);
    
    if (!powerSpectrum.empty()) {
        
        // grab index of max power spectrum
        double min, max;
        Point pmin, pmax;
        minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask);
        
        // calculate BPM
        double bpm = pmax.y * fps / total * SEC_PER_MIN;
        bpms.push_back(bpm);
        
        cout << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << " BPM=" << bpm << endl;
        
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
                    log << powerSpectrum.at<float>(i, 0) << "\n";
                }
            }
            log.close();
        }
        
        logfileDetailed << time << ";";
        logfileDetailed << bpm << "\n";
    }
    
    if ((time - lastSamplingTime) * timeBase >= samplingFrequency) {
        lastSamplingTime = time;
        
        cv::sort(bpms, bpms, SORT_EVERY_COLUMN);
        
        // average calculated BPMs since last sampling time
        meanBpm = mean(bpms)(0);
        minBpm = bpms.at<double>(0, 0);
        maxBpm = bpms.at<double>(bpms.rows-1, 0);
        
        std::cout << "meanBPM=" << meanBpm << std::endl;
        
        // Logging
        logfile << time << ";";
        logfile << meanBpm << ";";
        logfile << minBpm << ";";
        logfile << maxBpm << "\n";
        
        bpms.pop_back(bpms.rows);
    }
}

void RPPGSimple::draw(cv::Mat &frameRGB) {
    
    // Draw face shape
    ellipse(frameRGB,
            Point(box.tl().x + box.width / 2.0, box.tl().y + box.height / 2.0),
            Size(box.width / 2.5, box.height / 2.0),
            0, 0, 360, cv::GREEN);
    circle(frameRGB,
           Point(leftEye.tl().x + leftEye.width / 2.0, leftEye.tl().y + leftEye.height / 2.0),
           (leftEye.width + leftEye.height) / 4.0,
           cv::GREEN);
    circle(frameRGB,
           Point(rightEye.tl().x + rightEye.width / 2.0, rightEye.tl().y + rightEye.height / 2.0),
           (rightEye.width + rightEye.height) / 4.0,
           cv::GREEN);
    
    // Draw signal
    if (!signal.empty() && !powerSpectrum.empty()) {
        
        // Display of signals with fixed dimensions
        double displayHeight = box.height/2.0;
        double displayWidth = box.width*0.8;
        
        // Draw signal
        double vmin, vmax;
        Point pmin, pmax;
        minMaxLoc(signal, &vmin, &vmax, &pmin, &pmax);
        double heightMult = displayHeight/(vmax - vmin);
        double widthMult = displayWidth/(signal.rows - 1);
        double drawAreaTlX = box.tl().x + box.width;
        double drawAreaTlY = box.tl().y;
        Point p1(drawAreaTlX, drawAreaTlY + (vmax - signal.at<double>(0, 0))*heightMult);
        Point p2;
        for (int i = 1; i < signal.rows; i++) {
            p2 = Point(drawAreaTlX + i * widthMult, drawAreaTlY + (vmax - signal.at<double>(i, 0))*heightMult);
            line(frameRGB, p1, p2, RED, 2);
            p1 = p2;
        }
        
        // Draw powerSpectrum
        const int total = signal.rows;
        const int low = (int)(total * LOW_BPM / SEC_PER_MIN / fps);
        const int high = (int)(total * HIGH_BPM / SEC_PER_MIN / fps);
        Mat bandMask = Mat::zeros(signal.size(), CV_8U);
        bandMask.rowRange(min(low, total), min(high, total)).setTo(ONE);
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
    if (valid) {
        ss.precision(3);
        ss << meanBpm << " bpm";
        putText(frameRGB, ss.str(), Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_PLAIN, 2, cv::RED, 2);
    }
    
    // Draw FPS text
    ss.str("");
    ss << fps << " fps";
    putText(frameRGB, ss.str(), Point(box.tl().x, box.br().y + 40), cv::FONT_HERSHEY_PLAIN, 2, cv::GREEN, 2);
}