//
//  FFmpegDecoder.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef FFmpegDecoder_hpp
#define FFmpegDecoder_hpp

#include <stdio.h>
#include <string>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

class FFmpegDecoder {
    
    // Constructor
    public: FFmpegDecoder() : pImgConvertCtx(NULL), videoTimeBase(0.0), videoFramePerSecond(0.0), isOpen(false), videoStreamIndex(-1), pVideoCodec(NULL), pVideoCodecCtx(NULL), pFormatCtx(NULL) {;}
    
    // Destructor
    public: virtual ~FFmpegDecoder() {
        CloseFile();
    }
    
    // Open file
    public: virtual bool OpenFile(std::string &inputFile);
    
    // Close file and free resourses.
    public: virtual bool CloseFile();
    
    // Return next frame FFmpeg.
    public: virtual AVFrame * GetNextFrame();
    
    public: int GetWidth() {
        return width;
    }
    
    public: int GetHeight() {
        return height;
    }
    
    public: double GetFPS() {
        return videoFramePerSecond;
    }
    
    public: double GetTimeBase() {
        return videoTimeBase;
    }
    
    public: long GetStartTime() {
        return videoStartTime;
    }
    
    public: void FreeBuffer() {
        av_free(buffer);
    }
    
    // open video stream.
    private: bool OpenVideo();
    
    // close video stream.
    private: void CloseVideo();
    
    // return rgb image
    private: AVFrame * GetRGBAFrame(AVFrame *pFrameYuv);
    
    // Decode video buffer.
    private: bool DecodeVideo(const AVPacket *avpkt, AVFrame * pOutFrame);
    
    // FFmpeg file format.
    private: AVFormatContext* pFormatCtx;
    
    // FFmpeg codec context.
    private: AVCodecContext* pVideoCodecCtx;
    
    // FFmpeg codec for video.
    private: AVCodec* pVideoCodec;
    
    // Video stream number in file.
    private: int videoStreamIndex;
    
    // File is open or not.
    private: bool isOpen;
    
    // Video frame per seconds.
    private: double videoFramePerSecond;
    
    // FFmpeg timebase for video.
    private: double videoTimeBase;
    
    // FFmpeg Start time for video.
    private: long videoStartTime;
    
    // FFmpeg context convert image.
    private: struct SwsContext *pImgConvertCtx;
    
    // Width of image
    private: int width;
    
    // Height of image
    private: int height;
    
    // Buffer
    private: uint8_t * buffer;
    
};

#endif /* FFmpegDecoder_hpp */