//
//  FFmpegEncoder.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 26/03/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "FFmpegEncoder.hpp"
#include <iostream>

#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */

bool FFmpegEncoder::OpenFile(const char *filename, int width, int height, int bitrate, int framerate) {
        
    printf("Encode video file %s\n", filename);
    
    AVCodec *codec;
    
    /* Initialize libavcodec, and register all codecs and formats. */
    av_register_all();
    
    /* allocate the output media context */
    avformat_alloc_output_context2(&oc, NULL, NULL, filename);
    if (!oc) {
        printf("Could not deduce output format from file extension: using MPEG.\n");
        avformat_alloc_output_context2(&oc, NULL, "mpeg", filename);
    }
    if (!oc) {
        return false;
    }
    
    fmt = oc->oformat;
    
    /* Add the video stream using the default format codec
     * and initialize the codec. */
    
    st = NULL;
    
    if (fmt->video_codec != AV_CODEC_ID_NONE) {
        
        AVCodecContext *c;
        
        /* find the encoder */
        codec = avcodec_find_encoder(fmt->video_codec);
        if (!codec) {
            fprintf(stderr, "Codec not found\n");
            return false;
        }
        
        st = avformat_new_stream(oc, codec);
        if (!st) {
            fprintf(stderr, "Could not allocate stream\n");
            return false;
        }
        st->id = oc->nb_streams-1;
        c = st->codec;
        
        avcodec_get_context_defaults3(c, codec);
        c->codec_id = fmt->video_codec;
        c->bit_rate = bitrate;
        c->width    = width;
        c->height   = height;
        c->time_base.num = 1;
        c->time_base.den = framerate;
        c->ticks_per_frame = 1;
        //c->gop_size      = gopsize; /* emit one intra frame every x frames at most */
        c->pix_fmt       = STREAM_PIX_FMT;
    }
    
    /* Now that all the parameters are set, we can open the
     * video codec and allocate the necessary encode buffer. */
    
    if (st) {
        
        AVCodecContext *c = st->codec;
        
        // Open codec
        if (avcodec_open2(c, codec, NULL) < 0) {
            fprintf(stderr, "Could not open codec\n");
            return false;
        }
        
        /* Allocate the encoded raw picture. */
        dst = av_frame_alloc();
        if (!dst) {
            fprintf(stderr, "Could not allocate video frame\n");
            return false;
        }
        
        // Get image conversion context
        imgConvertCtx = sws_getContext(c->width, c->height, AV_PIX_FMT_BGR24, c->width, c->height, c->pix_fmt, SWS_BILINEAR, NULL, NULL, NULL);
        
        frame_count = 0;
        write_count = 0;
        buffer_count = 0;
    }
    
    av_dump_format(oc, 0, filename, 1);
    
    /* open the output file, if needed */
    if (!(fmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&oc->pb, filename, AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Could not open '%s'\n", filename);
            return false;
        }
    }
    
    /* Write the stream header, if any. */
    if (avformat_write_header(oc, NULL) < 0) {
        fprintf(stderr, "Error occurred when opening output file\n");
        return false;
    }
    
    return true;
}

void FFmpegEncoder::CloseFile() {
    
    av_write_trailer(oc);
    
    av_frame_free(&dst);
    avcodec_close(st->codec);
    avcodec_free_context(&st->codec);
    
    // Free streams
    for (int i = 0; i < oc->nb_streams; i++) {
        av_freep(&oc->streams[i]->codec);
        av_freep(&oc->streams[i]);
    }
    
    if (!(fmt->flags & AVFMT_NOFILE)) {
        /* Close the output file. */
        avio_close(oc->pb);
    }
    
    /* free the stream */
    avformat_free_context(oc);
}

void FFmpegEncoder::WriteFrame(AVFrame *frame, int64_t time) {
    
    std::cout << "Writing frame" << std::endl;
    
    int ret;
    AVCodecContext *c = st->codec;
    
    /* encode the image */
    AVPacket pkt;
    int got_output;
    
    av_init_packet(&pkt);
    pkt.data = NULL;    // packet data will be allocated by the encoder
    pkt.size = 0;
    
    /* Copy to dst in YUV format */
    int bufferImgSize = avpicture_get_size(c->pix_fmt, c->width, c->height);
    uint8_t *buffer = (uint8_t*)av_mallocz(bufferImgSize);
    avpicture_fill((AVPicture *)dst, buffer, c->pix_fmt, c->width, c->height);
    dst->format = STREAM_PIX_FMT;
    dst->width = c->width;
    dst->height = c->height;
    dst->pts = av_rescale_q(frame_count++, c->time_base, st->time_base);
    
    sws_scale(imgConvertCtx, frame->data, frame->linesize, 0, c->height, dst->data, dst->linesize);
    
    pts_queue.push(time);
    
    ret = avcodec_encode_video2(c, &pkt, dst, &got_output);
    if (ret < 0) {
        fprintf(stderr, "Error encoding video frame\n");
        exit(1);
    }
    
    /* If size is zero, it means the image was buffered. */
    if (got_output) {
        
        int64_t &pts = pts_queue.front();
        pts_queue.pop();
        
        pkt.pts = pts;
        pkt.dts = pts;
        //pkt.pts = av_rescale_q(pkt.pts, c->time_base, st->time_base);
        //pkt.dts = av_rescale_q(pkt.dts, c->time_base, st->time_base);
        
        if (c->coded_frame->key_frame)
            pkt.flags |= AV_PKT_FLAG_KEY;
        
        pkt.stream_index = st->index;
        
        /* Write the compressed frame to the media file. */
        ret = av_interleaved_write_frame(oc, &pkt);
        
        write_count++;
        
        std::cout << "GOT OUTPUT. WRITE COUNT: " << write_count << std::endl;
        
    } else {
        buffer_count++;
        std::cout << "NO OUTPUT. BUFFER COUNT: " << buffer_count << std::endl;
        ret = 0;
    }
    if (ret != 0) {
        fprintf(stderr, "Error while writing video frame\n");
        exit(1);
    }
}

void FFmpegEncoder::WriteBufferedFrames() {
    AVCodecContext *c = st->codec;
    for (int i = 0; i < buffer_count; i++) {
        AVPacket pkt;
        int got_output, ret;
        av_init_packet(&pkt);
        pkt.data = NULL;    // packet data will be allocated by the encoder
        pkt.size = 0;
        ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        if (ret < 0) {
            fprintf(stderr, "Error encoding video frame\n");
            exit(1);
        }
        if (got_output) {
            int64_t &pts = pts_queue.front();
            pts_queue.pop();
            pkt.pts = pts;
            pkt.dts = pts;
            if (c->coded_frame->key_frame)
                pkt.flags |= AV_PKT_FLAG_KEY;
            pkt.stream_index = st->index;
            /* Write the compressed frame to the media file. */
            ret = av_interleaved_write_frame(oc, &pkt);
            write_count++;
            std::cout << "GOT OUTPUT. WRITE COUNT: " << write_count << std::endl;
        }
    }
}