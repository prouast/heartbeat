# Heartbeat: Measuring heart rate using remote photoplethysmography (rPPG)

**News: Try out [VitalLens](https://apps.apple.com/us/app/vitallens/id6472757649), my new free iOS app implementing a modern rPPG model based on AI. We also have an [API](https://www.rouast.com/api/) with a [Python Client](https://github.com/Rouast-Labs/vitallens-python) which uses the same AI model. Find out more at [https://www.rouast.com/vitallens](https://www.rouast.com/vitallens/) and [https://www.rouast.com/api](https://www.rouast.com/api/).**

This is a simple implementation of rPPG, a way to measure heart rate without skin contact. It uses a video recording or live feed of the face to analyse subtle changes in skin color.

Here's how it works:

  - The face is detected and continuously tracked
  - Signal series is obtained by determining the facial color in every frame
  - Heart rate is estimated using frequency analysis and filtering of the series

If you are interested in the specifics, feel free to have a read of my publications on the topic:
  - [Remote Photoplethysmography: Evaluation of Contactless Heart Rate Measurement in an Information Systems Setting][aitic]
  - [Using Contactless Heart Rate Measurements for Real-Time Assessment of Affective States][gmunden]
  - [Remote heart rate measurement using low-cost RGB face video: A technical literature review][fcs]

See also my minimal [JavaScript implementation](https://github.com/prouast/heartbeat-js) and [Browser Demo](https://prouast.github.io/heartbeat-js/).

### Demo

* [Real-time rPPG in action][video1]
* [Offline rPPG With physiological baseline measurements][video2]

### Dependencies

The following libraries are required to run Heartbeat:

* [OpenCV]

They must be installed on the system such that headers and libraries are found on the compiler's standard search path.

### Installation

For building a Makefile is available that works on macOS:

```sh
$ make
```

Alternative compilation for Ubuntu. Works with opencv 3.1:

```sh
$ g++ -std=c++11 Heartbeat.cpp opencv.cpp RPPG.cpp `pkg-config --cflags --libs opencv` -o Heartbeat
```

### Settings

After building, the app can be run via

```
$ ./Heartbeat
```

Several command line arguments are available:

| Argument | Options | Description |
| --- | --- | --- |
| -i | Filepath to input video | Omit flag to use webcam |
| -rppg | g, pca (default: g) | Specify rPPG algorithm variant - only green channel or rgb channels with pca |
| -facedet | haar, deep (default: haar) | Specify face detection classifier - Haar cascade or deep neural network |
| -r | Re-detection interval (default: 1 s) | Interval for face re-detection; tracking is used frame-to-frame |
| -f | Sampling frequency (default: 1 Hz) | Frequency for heart rate estimation |
| -max | default: 5 | Maximum size of signal sliding window |
| -min | default: 5 | Minimum size of signal sliding window |
| -gui | true, false (default: true) | Display the GUI |
| -log | true, false (default: false) | Detailed logging |
| -ds | default: 1 | If using video from file: Downsample by using every ith frame |

License
----

GPL-3.0

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [aitic]: <https://www.rouast.com/pdf/rouast2016remote_b.pdf>
   [fcs]: <https://www.researchgate.net/profile/Raymond_Chiong/publication/306285292_Remote_heart_rate_measurement_using_low-cost_RGB_face_video_A_technical_literature_review/links/58098ac808ae1c98c252637d.pdf>
   [gmunden]: <http://link.springer.com/chapter/10.1007/978-3-319-41402-7_20>
   [OpenCV]: <http://opencv.org/downloads.html>
   [ffmpeg]: <https://ffmpeg.org/download.html>
   [video1]: <https://www.youtube.com/watch?v=D_KYv7pXAvQ>
   [video2]: <https://www.youtube.com/watch?v=4RKor-O5bQ8>
