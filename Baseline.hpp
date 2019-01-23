//
//  Baseline.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 17/09/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef Baseline_hpp
#define Baseline_hpp

#include <stdio.h>
#include <string>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

class Baseline {

public:

    // Constructor
    Baseline() {;}

    bool load(const double samplingFrequency, const double timeBase, const string baseline_path);

    // Process a frame
    void processFrame(Mat &frameRGB, int64_t time);

private:

    vector<string> readCSVRow(const string &row);

    // Settings
    double samplingFrequency;
    double timeBase;

    // Data
    vector<vector<string>> data;
    int dataIndex = 2;

    // State variables
    int64_t time;
    int64_t lastSamplingTime = 0;

    // Estimation
    double bpm_ppg = 0.0;
    double bpm_ecg = 0.0;
    vector<double> bpms_ppg;
    vector<double> bpms_ecg;
};

#endif /* Baseline_hpp */
