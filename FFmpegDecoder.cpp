//
//  FFmpegDecoder.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "FFmpegDecoder.hpp"
#include <iostream>

#define min(a,b) (a > b ? b : a)

bool FFmpegDecoder::OpenFile(std::string &inputFile) {
    
    CloseFile();
    
    // Register all formats and codecs
    av_register_all();
    
    // Open video file
    if (avformat_open_input(&pFormatCtx, inputFile.c_str(), NULL, NULL) != 0) {
        CloseFile();
        return false;
    }
    
    // Retrieve stream information
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
        CloseFile();
        return false;
    }
    
    // Open the video stream
    bool hasVideo = OpenVideo();
    if (!hasVideo) {
        CloseFile();
        return false;
    }
    
    isOpen = true;
    
    // Get file information.
    if (videoStreamIndex != -1) {
        videoStartTime      = pFormatCtx->streams[videoStreamIndex]->start_time;
        videoFramePerSecond = av_q2d(pFormatCtx->streams[videoStreamIndex]->r_frame_rate);
        // Need for convert time to ffmpeg time.
        videoTimeBase       = av_q2d(pFormatCtx->streams[videoStreamIndex]->time_base);
    }
    
    return true;
}

bool FFmpegDecoder::CloseFile() {
    isOpen = false;
    
    // Close video.
    CloseVideo();
    
    if (pFormatCtx) {
        avformat_close_input(&pFormatCtx);
        //av_close_input_file(pFormatCtx);
        pFormatCtx = NULL;
    }
    
    return true;
}

AVFrame * FFmpegDecoder::GetNextFrame() {
    
    AVFrame * res = NULL;
    
    if (videoStreamIndex != -1) {
        
        int frameFinished;
        
        AVPacket packet;
        av_init_packet(&packet);
        
        if (isOpen) {
            
            // Read packet.
            while (av_read_frame(pFormatCtx, &packet) >= 0) {
                
                int64_t pts = 0;
                
                AVFrame *pVideoYuv = av_frame_alloc();
                
                // Decode video frame
                avcodec_decode_video2(pVideoCodecCtx, pVideoYuv, &frameFinished, &packet);
                
                if (packet.dts != AV_NOPTS_VALUE) {
                    pts = av_frame_get_best_effort_timestamp(pVideoYuv);
                } else {
                    pts = 0;
                }
                
                if (frameFinished) {
                    res = GetRGBAFrame(pVideoYuv);
                    av_free(pVideoYuv);
                    av_frame_set_best_effort_timestamp(res, pts);
                    break;
                }
                
                av_free_packet(&packet);
            }
        }
    }
    
    return res;
}

AVFrame * FFmpegDecoder::GetRGBAFrame(AVFrame *pFrameYuv) {
    
    AVFrame *frame = av_frame_alloc();
    
    int width = pVideoCodecCtx->width;
    int height = pVideoCodecCtx->height;
    
    int bufferImgSize = avpicture_get_size(AV_PIX_FMT_BGR24, width, height);
    buffer = (uint8_t*)av_malloc(bufferImgSize);
    
    if (frame) {
        
        avpicture_fill((AVPicture*)frame, buffer, AV_PIX_FMT_BGR24, width, height);
        
        frame->width  = width;
        frame->height = height;
        
        sws_scale(pImgConvertCtx, pFrameYuv->data, pFrameYuv->linesize,
                  0, height, frame->data, frame->linesize);
    }
    
    return frame;
}

bool FFmpegDecoder::OpenVideo() {
    
    bool res = false;
    
    if (pFormatCtx) {
        
        videoStreamIndex = -1;
        
        // Find the first video stream
        for (unsigned int i = 0; i < pFormatCtx->nb_streams; i++) {
            
            if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
                
                videoStreamIndex = i;
                
                // Get a pointer to the codec context for the video stream
                pVideoCodecCtx = pFormatCtx->streams[i]->codec;
                
                // Find the decoder for the video stream
                pVideoCodec = avcodec_find_decoder(pVideoCodecCtx->codec_id);
                
                if (pVideoCodec) {
                    res     = !(avcodec_open2(pVideoCodecCtx, pVideoCodec, NULL) < 0);
                    width   = pVideoCodecCtx->coded_width;
                    height  = pVideoCodecCtx->coded_height;
                }
                
                break;
            }
        }
        
        if (!res) {
            CloseVideo();
        } else {
            pImgConvertCtx = sws_getContext(pVideoCodecCtx->width, pVideoCodecCtx->height,
                                            pVideoCodecCtx->pix_fmt,
                                            pVideoCodecCtx->width, pVideoCodecCtx->height,
                                            AV_PIX_FMT_BGR24,
                                            SWS_BICUBIC, NULL, NULL, NULL);
        }
    }
    
    return res;
}

bool FFmpegDecoder::DecodeVideo(const AVPacket *avpkt, AVFrame *pOutFrame) {

    bool res = false;
    
    if (pVideoCodecCtx) {
        
        if (avpkt && pOutFrame) {
            
            int got_picture_ptr = 0;
            int videoFrameBytes = avcodec_decode_video2(pVideoCodecCtx, pOutFrame, &got_picture_ptr, avpkt);
            
            res = (videoFrameBytes > 0);
        }
    }
    
    return res;
}

void FFmpegDecoder::CloseVideo() {
    
    if (pVideoCodecCtx) {
        
        avcodec_close(pVideoCodecCtx);
        pVideoCodecCtx = NULL;
        pVideoCodec = NULL;
        videoStreamIndex = 0;
    }
}

