# Heartbeat: Measuring heart rate using remote photoplethysmography (rPPG)

[![N|Solid](https://s12.postimg.org/gh7ymb02h/Logo.jpg)](https://github.com/prouast/heartbeat)

This is a simple implementation of rPPG, a way to measure heart rate without skin contact. It uses a video recording or live feed of the face to analyse subtle changes in skin color.

Here's how it works:

  - The face is detected and continuously tracked
  - Signal series is obtained by determining the facial color in every frame
  - Heart rate is estimated using frequency analysis and filtering of the series

If you are interested in the specifics, feel free to have a read of our publications on the topic:
  - [Remote Photoplethysmography: Evaluation of Contactless Heart Rate Measurement in an Information Systems Setting][aitic]
  - [Using Contactless Heart Rate Measurements for Real-Time Assessment of Affective States][gmunden]
  - [Remote heart rate measurement using low-cost RGB face video: A technical literature review][fcs]

### Demo

* [Real-time rPPG in action][video1]
* [Offline rPPG With physiological baseline measurements][video2]

### Dependencies

The following libraries are required to run Heartbeat:

* [OpenCV] 
* [ffmpeg]

* Note: On Ubuntu 16.04, it works with opencv 3.1

They must be installed on the system such that headers and libraries are found on the compiler's standard search path.

### Installation

Compile the source code for your system, providing a number of required linker flags, e.g.:

```sh
$ g++ -std=c++11 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_videoio -lavcodec -lavformat -lavutil -lswscale  Heartbeat.cpp FFmpegDecoder.cpp FFmpegEncoder.cpp opencv.cpp RPPG.cpp Baseline.cpp -o Heartbeat
```

Alternative compilation for Ubuntu
```sh
$ g++ -std=c++11 Heartbeat.cpp FFmpegDecoder.cpp FFmpegEncoder.cpp opencv.cpp RPPG.cpp Baseline.cpp `pkg-config --cflags --libs opencv` -lavcodec -lavformat -lavutil -lswscale  -o Heartbeat
```

License
----

GPL-3.0

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [aitic]: <http://air.newcastle.edu.au/AITIC_files/Paper_40.pdf>
   [fcs]: <https://www.researchgate.net/profile/Raymond_Chiong/publication/306285292_Remote_heart_rate_measurement_using_low-cost_RGB_face_video_A_technical_literature_review/links/58098ac808ae1c98c252637d.pdf>
   [gmunden]: <http://link.springer.com/chapter/10.1007/978-3-319-41402-7_20>
   [OpenCV]: <http://opencv.org/downloads.html>
   [ffmpeg]: <https://ffmpeg.org/download.html>
   [video1]: <https://www.youtube.com/watch?v=D_KYv7pXAvQ>
   [video2]: <https://www.youtube.com/watch?v=4RKor-O5bQ8>
