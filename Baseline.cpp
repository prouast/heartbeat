//
//  Baseline.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 17/09/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "Baseline.hpp"

#include <fstream>
#include "opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>

bool Baseline::load(const double samplingFrequency, const double timeBase, const string baseline_path) {

    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;

    std::ifstream fileStream(baseline_path);
    string row;

    while (true) {
        getline(fileStream, row);
        if (fileStream.bad() || fileStream.eof()) {
            break;
        }
        auto fields = readCSVRow(row);
        data.push_back(fields);
    }

    return true;
}

void Baseline::processFrame(Mat &frameRGB, int64_t time) {

    // Set time
    this->time = time + 1466005435646000;

    cout << data[dataIndex][1] << " vs " << this->time << endl;

    // Read new values in buffer
    while (stol(data[dataIndex][1]) <= this->time) {
        bpms_ppg.push_back(atof(data[dataIndex][2].c_str()));
        bpms_ecg.push_back(atof(data[dataIndex][3].c_str()));
        dataIndex++;
    }

    cout << bpms_ppg.size() << " di=" << dataIndex << endl;

    if ((time - lastSamplingTime) * timeBase >= 1/samplingFrequency) {
        lastSamplingTime = time;

        bpm_ppg = mean(bpms_ppg)(0);
        bpm_ecg = mean(bpms_ecg)(0);

        bpms_ppg.clear();
        bpms_ecg.clear();
    }

    // Draw PPG
    std::stringstream ss;
    ss.precision(3);
    ss << "PPG baseline: " << bpm_ppg << " bpm";
    cv::putText(frameRGB, ss.str(), Point(frameRGB.cols - 400, frameRGB.rows - 30), FONT_HERSHEY_PLAIN, 2, RED, 2);

    // Draw PPG
    ss.str("");
    ss << "ECG baseline: " << bpm_ecg << " bpm";
    cv::putText(frameRGB, ss.str(), Point(frameRGB.cols - 400, frameRGB.rows - 10), FONT_HERSHEY_PLAIN, 2, RED, 2);
}

vector<string> Baseline::readCSVRow(const std::string &row) {
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; // index of the current field
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': // end of field
                        fields.push_back(""); i++;
                        break;
                    case '"': state = CSVState::QuotedField;
                        break;
                    default:  fields[i].push_back(c);
                    break; }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': state = CSVState::QuotedQuote;
                        break;
                    default:  fields[i].push_back(c);
                    break; }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': // , after closing quote
                        fields.push_back(""); i++;
                        state = CSVState::UnquotedField;
                        break;
                    case '"': // "" -> "
                        fields[i].push_back('"');
                        state = CSVState::QuotedField;
                        break;
                    default:  // end of quote
                        state = CSVState::UnquotedField;
                    break; }
                break;
        }
    }
    return fields;
}
