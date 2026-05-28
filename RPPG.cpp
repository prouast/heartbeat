//
//  RPPG.cpp
//  Heartbeat
//
//  Create by Neel Ganesh 11/26/25`
//

#include "RPPG.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "opencv.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

bool RPPG::load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
                const int width, const int height, const double timeBase, const int downsample,
                const double samplingFrequency, const double rescanFrequency,
                const int minSignalSize, const int maxSignalSize,
                const string &logPath, const string &haarPath,
                const string &dnnProtoPath, const string &dnnModelPath,
                const bool log, const bool gui) {

    this->rPPGAlg = rPPGAlg;
    this->faceDetAlg = faceDetAlg;
    this->guiMode = gui;
    this->lastSamplingTime = 0;
    this->logMode = log;
    this->minFaceSize = Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
    this->maxSignalSize = maxSignalSize;
    this->minSignalSize = minSignalSize;
    this->rescanFlag = false;
    this->rescanFrequency = rescanFrequency;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;

    // Load classifier
    switch (faceDetAlg) {
      case haar:
        haarClassifier.load(haarPath);
        break;
      case deep:
        dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
        break;
    }

    // Setting up logfilepath
    ostringstream path_1;
    path_1 << logPath << "_rppg=" << rPPGAlg << "_facedet=" << faceDetAlg << "_min=" << minSignalSize << "_max=" << maxSignalSize << "_ds=" << downsample;
    this->logfilepath = path_1.str();

    // Logging bpm according to sampling frequency
    std::ostringstream path_2;
    path_2 << logfilepath << "_bpm.csv";
    logfile.open(path_2.str());
    logfile << "time;face_valid;mean;min;max\n";
    logfile.flush();

    // Logging bpm detailed (includes smoothed BPM for spreadsheet analysis)
    std::ostringstream path_3;
    path_3 << logfilepath << "_bpmAll.csv";
    logfileDetailed.open(path_3.str());
    logfileDetailed << "time;face_valid;bpm;smoothed_bpm\n";
    logfileDetailed.flush();

    return true;
}

void RPPG::exit() {
    logfile.close();
    logfileDetailed.close();
}

void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, int time) {

    // Set time
    this->time = time;
    
    // Invalidate stale aggregate averages after 5 seconds.
    if (lastSamplingTime > 0 &&
        ((time - lastSamplingTime) * timeBase) > AVERAGE_VALIDITY_SECONDS) {
        meanBpm = 0.0;
        minBpm = 0.0;
        maxBpm = 0.0;
    }

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
        while (s.rows > fps * maxSignalSize) {
            push(s);
            push(t);
            push(re);
        }

        assert(s.rows == t.rows && s.rows == re.rows);

        // New values
        Scalar means = mean(frameRGB, mask);
        // Add new values to raw signal buffer
        double values[] = {means(0), means(1), means(2)};
        s.push_back(Mat(1, 3, CV_64F, values));
        t.push_back(time);

        // Save rescan flag
        re.push_back(rescanFlag);

        // Update fps
        fps = getFps(t, timeBase);

        // Update band spectrum limits
        low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
        high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1;

        // If valid signal is large enough: estimate
        if (s.rows >= fps * minSignalSize) {

            // Filtering
            switch (rPPGAlg) {
                case g:
                    extractSignal_g();
                    break;
                case pca:
                    extractSignal_pca();
                    break;
                case xminay:
                    extractSignal_xminay();
                    break;
            }

            // HR estimation
            estimateHeartrate();
            
            // Update signal history for waveform display
            if (!s_f.empty()) {
                double latestSignal = s_f.at<double>(s_f.rows - 1, 0);
                signalHistory.push_back(latestSignal);
                if (signalHistory.size() > MAX_HISTORY_SIZE) {
                    signalHistory.erase(signalHistory.begin());
                }
            }

            // Log
            log();
        }

        if (guiMode) {
            draw(frameRGB);
        }
    }

    rescanFlag = false;

    frameGray.copyTo(lastFrameGray);
}

void RPPG::detectFace(Mat &frameRGB, Mat &frameGray) {

    cout << "Scanning for faces…" << endl;
    vector<Rect> boxes = {};

    switch (faceDetAlg) {
      case haar:
        // Detect faces with Haar classifier
        haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
        break;
      case deep:
        // Detect faces with DNN
        Mat resize300;
        cv::resize(frameRGB, resize300, Size(300, 300));
        Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        dnnClassifier.setInput(blob);
        Mat detection = dnnClassifier.forward();
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        float confidenceThreshold = 0.5;

        for (int i = 0; i < detectionMat.rows; i++) {
          float confidence = detectionMat.at<float>(i, 2);
          if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            boxes.push_back(object);
          }
        }
        break;
    }

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

void RPPG::setNearestBox(vector<Rect> boxes) {
    int index = 0;
    Point p = box.tl() - boxes.at(0).tl();
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

void RPPG::detectCorners(Mat &frameGray) {

    // Define tracking region
    Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
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
    fillPoly(trackingRegion, pts, npts, 1, WHITE);

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

void RPPG::trackFace(Mat &frameGray) {

    // Make sure enough corners are available
    if (corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }

    Contour2f corners_1;
    Contour2f corners_0;
    vector<uchar> cornersFound_1;
    vector<uchar> cornersFound_0;
    Mat err;

    // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
    calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

    // Backtrack once to make it more robust
    calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

    // Exclude no-good corners
    Contour2f corners_1v;
    Contour2f corners_0v;
    for (size_t j = 0; j < corners.size(); j++) {
        if (cornersFound_1[j] && cornersFound_0[j]
            && norm(corners[j]-corners_0[j]) < 2) {
            corners_0v.push_back(corners_0[j]);
            corners_1v.push_back(corners_1[j]);
        } else {
            cout << "Mis!" << std::endl;
        }
    }

    if (corners_1v.size() >= MIN_CORNERS) {

        // Save updated features
        corners = corners_1v;

        // Estimate affine transform
        Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);

        if (transform.total() > 0) {

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
        }

    } else {
        cout << "Tracking failed! Not enough corners left." << endl;
        invalidateFace();
    }
}

void RPPG::updateROI() {
    this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
                     Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height));
}

void RPPG::updateMask(Mat &frameGray) {

    cout << "Update mask" << endl;

    mask = Mat::zeros(frameGray.size(), frameGray.type());
    rectangle(mask, this->roi, WHITE, FILLED);
}

void RPPG::invalidateFace() {

    s = Mat1d();
    s_f = Mat1d();
    t = Mat1d();
    re = Mat1b();
    powerSpectrum = Mat1d();
    faceValid = false;
}

void RPPG::extractSignal_g() {

    // Denoise - optimized for accuracy
    Mat s_den = Mat(s.rows, 1, CV_64F);
    denoise(s.col(1), re, s_den);

    // Normalise
    normalization(s_den, s_den);

    // Detrend - optimized lambda for better accuracy
    Mat s_det = Mat(s_den.rows, s_den.cols, CV_64F);
    int lambda = static_cast<int>(fps * 1.0);  // Use full fps for better detrending
    if (lambda < 1) lambda = 1;
    detrend(s_den, s_det, lambda);

    // Apply bandpass filter with optimized parameters for better accuracy
    Mat s_band = Mat(s_det.rows, s_det.cols, CV_64F);
    if (s_det.rows > 10 && low >= 0 && high > low && high < s_det.rows) {
        try {
            bandpass(s_det, s_band, low, high);
        } catch (...) {
            s_det.copyTo(s_band);
        }
    } else {
        s_det.copyTo(s_band);
    }

    // Moving average - optimized window size for better accuracy
    Mat s_mav = Mat(s_band.rows, s_band.cols, CV_64F);
    int maWindow = fmax(floor(fps/8), 1);  // Smaller window = more responsive
    movingAverage(s_band, s_mav, 3, maWindow);

    s_mav.copyTo(s_f);

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "re;g;g_den;g_det;g_mav\n";
        for (int i = 0; i < s.rows; i++) {
            log << re.at<bool>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_det.at<double>(i, 0) << ";";
            log << s_mav.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::extractSignal_pca() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize signals
    normalization(s_den, s_den);

    // Detrend
    Mat s_det = Mat(s.rows, s.cols, CV_64F);
    detrend(s_den, s_det, fps);

    // PCA to reduce dimensionality
    Mat s_pca = Mat(s.rows, 1, CV_32F);
    Mat pc = Mat(s.rows, s.cols, CV_32F);
    pcaComponent(s_det, s_pca, pc, low, high);

    // Moving average
    Mat s_mav = Mat(s.rows, 1, CV_32F);
    movingAverage(s_pca, s_mav, 3, fmax(floor(fps/6), 2));

    s_mav.copyTo(s_f);

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "re;r;g;b;r_den;g_den;b_den;r_det;g_det;b_det;pc1;pc2;pc3;s_pca;s_mav\n";
        for (int i = 0; i < s.rows; i++) {
            log << re.at<bool>(i, 0) << ";";
            log << s.at<double>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s.at<double>(i, 2) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_den.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 2) << ";";
            log << s_det.at<double>(i, 0) << ";";
            log << s_det.at<double>(i, 1) << ";";
            log << s_det.at<double>(i, 2) << ";";
            log << pc.at<double>(i, 0) << ";";
            log << pc.at<double>(i, 1) << ";";
            log << pc.at<double>(i, 2) << ";";
            log << s_pca.at<double>(i, 0) << ";";
            log << s_mav.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::extractSignal_xminay() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize raw signals
    Mat s_n = Mat(s_den.rows, s_den.cols, CV_64F);
    normalization(s_den, s_n);

    // Calculate X_s signal
    Mat x_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);

    // Calculate Y_s signal
    Mat y_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
    addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);

    // Bandpass
    Mat x_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(x_s, x_f, low, high);
    x_f.convertTo(x_f, CV_64F);
    Mat y_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(y_s, y_f, low, high);
    y_f.convertTo(y_f, CV_64F);

    // Calculate alpha
    Scalar mean_x_f;
    Scalar stddev_x_f;
    meanStdDev(x_f, mean_x_f, stddev_x_f);
    Scalar mean_y_f;
    Scalar stddev_y_f;
    meanStdDev(y_f, mean_y_f, stddev_y_f);
    double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];

    // Calculate signal
    Mat xminay = Mat(s.rows, 1, CV_64F);
    addWeighted(x_f, 1, y_f, -alpha, 0, xminay);

    // Moving average
    movingAverage(xminay, s_f, 3, fmax(floor(fps/6), 2));

    // Logging
    if (logMode) {
        std::ofstream log;
        std::ostringstream filepath;
        filepath << logfilepath << "_signal_" << time << ".csv";
        log.open(filepath.str());
        log << "r;g;b;r_den;g_den;b_den;x_s;y_s;x_f;y_f;s;s_f\n";
        for (int i = 0; i < s.rows; i++) {
            log << s.at<double>(i, 0) << ";";
            log << s.at<double>(i, 1) << ";";
            log << s.at<double>(i, 2) << ";";
            log << s_den.at<double>(i, 0) << ";";
            log << s_den.at<double>(i, 1) << ";";
            log << s_den.at<double>(i, 2) << ";";
            log << x_s.at<double>(i, 0) << ";";
            log << y_s.at<double>(i, 0) << ";";
            log << x_f.at<double>(i, 0) << ";";
            log << y_f.at<double>(i, 0) << ";";
            log << xminay.at<double>(i, 0) << ";";
            log << s_f.at<double>(i, 0) << "\n";
        }
        log.close();
    }
}

void RPPG::estimateHeartrate() {
    powerSpectrum = cv::Mat(s_f.size(), CV_32F);
    timeToFrequency(s_f, powerSpectrum, true);

    // band mask
    const int total = s_f.rows;
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
        
        // Outlier detection with last-reading trend awareness (no moving averages).
        bool isValidBpm = (bpm >= MIN_REASONABLE_BPM && bpm <= MAX_REASONABLE_BPM);
        bool isTrendValid = true;
        
        // Check for sudden changes against last accepted reading only.
        if (bpmHistory.size() > 0) {
            const double lastValidBpm = bpmHistory.back();
            const double change = bpm - lastValidBpm;
            
            // Be very suspicious of sudden drops, more lenient with rises
            if (change < -MAX_BPM_DROP_PER_READING) {
                isTrendValid = false;
                cout << "Rejected sudden BPM drop: " << bpm << " (dropped " << abs(change) << " from last " << lastValidBpm << ")" << endl;
            } else if (change > MAX_BPM_RISE_PER_READING) {
                isTrendValid = false;
                cout << "Rejected sudden BPM spike: " << bpm << " (rose " << change << " from last " << lastValidBpm << ")" << endl;
            }
        }
        
        // Only add valid BPM values to history for smoothing
        if (isValidBpm && isTrendValid) {
            bpmHistory.push_back(bpm);
            if (bpmHistory.size() > BPM_HISTORY_SIZE) {
                bpmHistory.erase(bpmHistory.begin());
            }
        } else {
            // Reject outlier - don't add to history
            if (!isValidBpm) {
                cout << "Rejected outlier BPM: " << bpm << " (outside range " << MIN_REASONABLE_BPM << "-" << MAX_REASONABLE_BPM << ")" << endl;
            }
        }
        
        // Calculate smoothed BPM with very light smoothing so natural fluctuations remain visible.
        if (bpmHistory.size() >= 3) {
            const double latestValidBpm = bpmHistory.back();
            
            // Light 30/70 blend keeps anti-jitter behavior without flattening real movement.
            if (smoothedBpm > 0 && latestValidBpm > 0) {
                double proposedSmooth = smoothedBpm * 0.30 + latestValidBpm * 0.70;
                
                // Additional safety: never allow smoothed value to drop below threshold
                if (proposedSmooth >= MIN_REASONABLE_BPM) {
                    smoothedBpm = proposedSmooth;
                } else {
                    // Keep previous smoothed value if new would be too low
                    cout << "Preserving previous smoothed BPM " << smoothedBpm << " (proposed " << proposedSmooth << " too low)" << endl;
                }
            } else if (latestValidBpm >= MIN_REASONABLE_BPM) {
                smoothedBpm = latestValidBpm;
            }
            
            // Stage 4: Fallback protection - if raw bpm is outlier but smoothed is good, use smoothed
            if ((!isValidBpm || !isTrendValid) && smoothedBpm >= MIN_REASONABLE_BPM && smoothedBpm <= MAX_REASONABLE_BPM) {
                bpm = smoothedBpm;  // Override outlier with stable smoothed value
                cout << "Using smoothed BPM " << smoothedBpm << " instead of outlier " << (pmax.y * fps / total * SEC_PER_MIN) << endl;
            }
        } else if (bpmHistory.size() > 0) {
            // No averaging: use the latest valid reading.
            smoothedBpm = bpmHistory.back();
        }
        
        // Enforce sane bounds and keep BPM floor in requested resting range.
        if (smoothedBpm > 0) {
            if (smoothedBpm < MIN_REASONABLE_BPM) smoothedBpm = MIN_REASONABLE_BPM;
            if (smoothedBpm > MAX_REASONABLE_BPM) smoothedBpm = MAX_REASONABLE_BPM;
        }
        if (bpm > 0) {
            if (bpm < MIN_REASONABLE_BPM) bpm = MIN_REASONABLE_BPM;
            if (bpm > MAX_REASONABLE_BPM) bpm = MAX_REASONABLE_BPM;
        }

        cout << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << " BPM=" << bpm << " Smoothed=" << smoothedBpm << endl;

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

        // average calculated BPMs since last sampling time
        meanBpm = mean(bpms)(0);
        minBpm = bpms.at<double>(0, 0);
        maxBpm = bpms.at<double>(bpms.rows-1, 0);

        std::cout << "meanBPM=" << meanBpm << " minBpm=" << minBpm << " maxBpm=" << maxBpm << std::endl;

        bpms.pop_back(bpms.rows);
    }
}

void RPPG::log() {

    if (lastSamplingTime == time || lastSamplingTime == 0) {
        logfile << time << ";";
        logfile << faceValid << ";";
        logfile << meanBpm << ";";
        logfile << minBpm << ";";
        logfile << maxBpm << "\n";
        logfile.flush();
    }

    logfileDetailed << time << ";";
    logfileDetailed << faceValid << ";";
    logfileDetailed << bpm << ";";
    logfileDetailed << smoothedBpm << "\n";
    logfileDetailed.flush();
}

void RPPG::draw(cv::Mat &frameRGB) {
    if (frameRGB.empty()) return;
    
    const int frameWidth = frameRGB.cols;
    const int frameHeight = frameRGB.rows;
    const int padding = 20;
    
    // Minimum frame size check - skip UI if frame is too small to avoid crashes
    const int MIN_FRAME_WIDTH = 400;
    const int MIN_FRAME_HEIGHT = 200;
    if (frameWidth < MIN_FRAME_WIDTH || frameHeight < MIN_FRAME_HEIGHT) {
        // Fallback: Just draw basic text info for small frames
        std::stringstream ss;
        if (faceValid && smoothedBpm > 0) {
            ss << std::fixed << std::setprecision(0) << smoothedBpm << " BPM";
            putText(frameRGB, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 100), 2);
        } else {
            putText(frameRGB, "No Signal", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(100, 100, 100), 2);
        }
        std::string statusText = faceValid ? "Tracking" : "Searching...";
        putText(frameRGB, statusText, Point(10, frameHeight - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
        return;
    }
    
    // Modern UI: Corner overlays and floating elements that don't cover the face
    // Top-left: Large BPM display
    // Top-right: Waveform graph (compact)
    // Bottom-left: Status info
    // Bottom-right: Quality indicator
    
    // Use smoothed BPM for display - always prefer smoothed for stable display
    // This eliminates fluctuations from raw readings
    double displayBpm = 0;
    if (smoothedBpm > 0 && smoothedBpm >= MIN_REASONABLE_BPM && smoothedBpm <= MAX_REASONABLE_BPM) {
        // Always use smoothed BPM if available and valid (most stable)
        displayBpm = smoothedBpm;
    } else if (bpm > 0 && bpm >= MIN_REASONABLE_BPM && bpm <= MAX_REASONABLE_BPM) {
        // Fallback to current bpm if valid
        displayBpm = bpm;
    }
    
    // === TOP-LEFT: Large BPM Display ===
    int bpmX = padding;
    int bpmY = padding + 60;
    int bpmWidth = 280;
    int bpmHeight = 140;
    
    // Bounds checking for BPM rectangle
    int bpmRectX = bpmX;
    int bpmRectY = std::max(0, bpmY - 50);
    int bpmRectWidth = std::min(bpmWidth, frameWidth - bpmRectX - 5);
    int bpmRectHeight = std::min(bpmHeight, frameHeight - bpmRectY - 5);
    
    // Only draw if rectangle is valid
    if (bpmRectWidth > 50 && bpmRectHeight > 50) {
        // Semi-transparent background with rounded corners effect
        Rect bpmRect(bpmRectX, bpmRectY, bpmRectWidth, bpmRectHeight);
        Mat bpmOverlay = frameRGB(bpmRect).clone();
        bpmOverlay.setTo(Scalar(20, 20, 25));
        addWeighted(frameRGB(bpmRect), 0.3, bpmOverlay, 0.7, 0, frameRGB(bpmRect));
        
        // Border
        rectangle(frameRGB, bpmRect, Scalar(100, 150, 255), 2);
        
        // Heart icon
        circle(frameRGB, Point(bpmX + 25, bpmY - 25), 12, Scalar(255, 50, 50), -1);
        
        // "HEART RATE" label
        putText(frameRGB, "HEART RATE", Point(bpmX + 50, bpmY - 15), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        
        // Large BPM number
        std::stringstream ss;
        if (faceValid && displayBpm > 0) {
            ss << std::fixed << std::setprecision(0) << displayBpm;
            Scalar bpmColor = Scalar(0, 255, 100);  // Green for good signal
            if (displayBpm < 50 || displayBpm > 120) {
                bpmColor = Scalar(100, 200, 255);  // Blue for unusual values
            }
            putText(frameRGB, ss.str(), Point(bpmX + 20, bpmY + 40), 
                   FONT_HERSHEY_SIMPLEX, 2.8, bpmColor, 4);
            putText(frameRGB, "BPM", Point(bpmX + 20, bpmY + 75), 
                   FONT_HERSHEY_SIMPLEX, 0.9, Scalar(150, 150, 150), 2);
        } else {
            putText(frameRGB, "--", Point(bpmX + 20, bpmY + 40), 
                   FONT_HERSHEY_SIMPLEX, 2.8, Scalar(100, 100, 100), 4);
            putText(frameRGB, "No Signal", Point(bpmX + 20, bpmY + 75), 
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 100), 2);
        }
    }
    
    // === TOP-RIGHT: Compact Waveform Graph ===
    int graphX = std::max(padding, frameWidth - 320 - padding);
    int graphY = padding + 20;
    int graphWidth = 300;
    int graphHeight = 100;
    
    // Bounds checking for graph rectangle
    int graphRectX = std::max(0, graphX);
    int graphRectY = std::max(0, graphY);
    int graphRectWidth = std::min(graphWidth, frameWidth - graphRectX - 5);
    int graphRectHeight = std::min(graphHeight, frameHeight - graphRectY - 5);
    
    // Only draw if rectangle is valid
    if (graphRectWidth > 100 && graphRectHeight > 50) {
        // Background
        Rect graphRect(graphRectX, graphRectY, graphRectWidth, graphRectHeight);
        Mat graphOverlay = frameRGB(graphRect).clone();
        graphOverlay.setTo(Scalar(15, 15, 20));
        addWeighted(frameRGB(graphRect), 0.4, graphOverlay, 0.6, 0, frameRGB(graphRect));
        rectangle(frameRGB, graphRect, Scalar(60, 60, 80), 2);
        
        // Graph title
        putText(frameRGB, "Pulse Waveform", Point(graphRectX + 10, graphRectY - 5), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        
        // Draw waveform if available
        if (signalHistory.size() > 1 && faceValid) {
            size_t startIdx = (signalHistory.size() > 150) ? signalHistory.size() - 150 : 0;
            size_t numPoints = signalHistory.size() - startIdx;
            
            if (numPoints > 1) {
                double vmin = *min_element(signalHistory.begin() + startIdx, signalHistory.end());
                double vmax = *max_element(signalHistory.begin() + startIdx, signalHistory.end());
                double range = vmax - vmin;
                if (range < 0.001) range = 0.001;
                
                double widthStep = (double)graphRectWidth / (numPoints - 1);
                double heightMult = (graphRectHeight - 10) / range;
                double centerY = graphRectY + graphRectHeight / 2.0;
                
                // Center line
                line(frameRGB, Point(graphRectX + 5, centerY), Point(graphRectX + graphRectWidth - 5, centerY), 
                     Scalar(40, 40, 50), 1);
                
                // Waveform
                int step = (numPoints > 80) ? 2 : 1;
                Point p1(graphRectX + 5, centerY - (signalHistory[startIdx] - (vmin + vmax)/2.0) * heightMult);
                for (size_t i = step; i < numPoints; i += step) {
                    double x = graphRectX + 5 + i * widthStep;
                    double y = centerY - (signalHistory[startIdx + i] - (vmin + vmax)/2.0) * heightMult;
                    Point p2(x, y);
                    line(frameRGB, p1, p2, Scalar(0, 255, 100), 2);  // Green pulse line
                    p1 = p2;
                }
            }
        } else {
            putText(frameRGB, "Waiting...", Point(graphRectX + graphRectWidth/2 - 40, graphRectY + graphRectHeight/2), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 100), 1);
        }
    }
    
    // === BOTTOM-LEFT: Status Info ===
    int statusX = padding;
    int statusY = std::max(padding, frameHeight - 120);
    
    // Bounds checking for status rectangle
    int statusRectX = statusX;
    int statusRectY = std::max(0, statusY);
    int statusRectWidth = std::min(200, frameWidth - statusRectX - 5);
    int statusRectHeight = std::min(100, frameHeight - statusRectY - 5);
    
    // Only draw if rectangle is valid
    if (statusRectWidth > 100 && statusRectHeight > 40) {
        // Background
        Rect statusRect(statusRectX, statusRectY, statusRectWidth, statusRectHeight);
        Mat statusOverlay = frameRGB(statusRect).clone();
        statusOverlay.setTo(Scalar(20, 20, 25));
        addWeighted(frameRGB(statusRect), 0.3, statusOverlay, 0.7, 0, frameRGB(statusRect));
        rectangle(frameRGB, statusRect, Scalar(60, 60, 80), 2);
        
        // Status text
        std::string statusText = faceValid ? "Tracking" : "Searching...";
        Scalar statusColor = faceValid ? Scalar(0, 255, 100) : Scalar(255, 200, 100);
        putText(frameRGB, "Status:", Point(statusRectX + 10, statusRectY + 25), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        putText(frameRGB, statusText, Point(statusRectX + 10, statusRectY + 50), 
               FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
        
        // FPS
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << fps << " fps";
        putText(frameRGB, ss.str(), Point(statusRectX + 10, statusRectY + 75), 
               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 150, 150), 1);
    }
    
    // === BOTTOM-RIGHT: Quality Indicator ===
    int qualityX = std::max(padding, frameWidth - 180 - padding);
    int qualityY = std::max(padding, frameHeight - 100);
    
    // Bounds checking for quality rectangle
    int qualityRectX = std::max(0, qualityX);
    int qualityRectY = std::max(0, qualityY);
    int qualityRectWidth = std::min(180, frameWidth - qualityRectX - 5);
    int qualityRectHeight = std::min(80, frameHeight - qualityRectY - 5);
    
    // Only draw if rectangle is valid
    if (qualityRectWidth > 100 && qualityRectHeight > 40) {
        // Background
        Rect qualityRect(qualityRectX, qualityRectY, qualityRectWidth, qualityRectHeight);
        Mat qualityOverlay = frameRGB(qualityRect).clone();
        qualityOverlay.setTo(Scalar(20, 20, 25));
        addWeighted(frameRGB(qualityRect), 0.3, qualityOverlay, 0.7, 0, frameRGB(qualityRect));
        rectangle(frameRGB, qualityRect, Scalar(60, 60, 80), 2);
        
        // Quality bar
        if (faceValid && bpmHistory.size() > 5) {
            // Calculate stability from BPM history variance
            double meanBpmHist = 0;
            for (double b : bpmHistory) meanBpmHist += b;
            meanBpmHist /= bpmHistory.size();
            
            double variance = 0;
            for (double b : bpmHistory) variance += (b - meanBpmHist) * (b - meanBpmHist);
            variance /= bpmHistory.size();
            double stddev = sqrt(variance);
            
            // Quality based on low variance (more stable = better)
            double quality = std::max(0.0, std::min(1.0, 1.0 - (stddev / 10.0)));
            
            putText(frameRGB, "Signal Quality", Point(qualityRectX + 10, qualityRectY + 20), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            
            // Quality bar (adapt to available width)
            int barWidth = std::min(160, qualityRectWidth - 20);
            int barHeight = 15;
            Rect barRect(qualityRectX + 10, qualityRectY + 35, barWidth, barHeight);
            rectangle(frameRGB, barRect, Scalar(40, 40, 50), -1);
            rectangle(frameRGB, barRect, Scalar(100, 100, 100), 1);
            
            int fillWidth = (int)(barWidth * quality);
            if (fillWidth > 0) {
                Scalar barColor = Scalar(0, 255, 100);  // Green
                if (quality < 0.5) barColor = Scalar(255, 200, 100);  // Yellow
                if (quality < 0.3) barColor = Scalar(100, 100, 255);  // Red
                Rect fillRect(qualityRectX + 10, qualityRectY + 35, fillWidth, barHeight);
                rectangle(frameRGB, fillRect, barColor, -1);
            }
            
            // Percentage
            std::stringstream ss;
            ss << std::fixed << std::setprecision(0) << (quality * 100) << "%";
            putText(frameRGB, ss.str(), Point(qualityRectX + 10, qualityRectY + 65), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 150, 150), 1);
        } else {
            putText(frameRGB, "Signal Quality", Point(qualityRectX + 10, qualityRectY + 20), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            putText(frameRGB, "Calculating...", Point(qualityRectX + 10, qualityRectY + 45), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 100), 1);
        }
    }
}
