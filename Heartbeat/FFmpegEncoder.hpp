//
//  FFmpegEncoder.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 26/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef FFmpegEncoder_hpp
#define FFmpegEncoder_hpp

#include <stdio.h>
#include <string>
#include <queue>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class FFmpegEncoder {
    
    
public:
    
    // Constructor
    FFmpegEncoder() : fmt(NULL), oc(NULL), st(NULL), imgConvertCtx(NULL), dst(NULL) {;}
    
    // Open file
    bool OpenFile(const char *filename, int width, int height, int bitrate, int framerate);
    
    // Write next frame.
    void WriteFrame(AVFrame *frame, int64_t time);
    
    // Write buffered frames.
    void WriteBufferedFrames();
    
    // Close file and free resourses.
    void CloseFile();
    
private:
    
    int64_t frame_count;
    int64_t write_count;
    int64_t buffer_count;
    std::queue<int64_t> pts_queue;
    
    AVFrame *dst;
    
    AVOutputFormat *fmt;                //
    AVFormatContext* oc;                //
    AVStream *st;                       // FFmpeg stream
    struct SwsContext *imgConvertCtx;   // FFmpeg context convert image.
};

#endif /* FFmpegEncoder_hpp */
