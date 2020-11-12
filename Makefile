# Makefile for heartbeat
appname := Heartbeat

CXX := g++
RM := rm -f
CXXFLAGS := -Wall -g -std=c++11 -I/usr/local/include/opencv4 -I/usr/include/opencv4
LDFLAGS := -g
LDLIBS := -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_videoio

SRCS := $(shell find . -name "*.cpp")
OBJS = $(subst .cpp,.o,$(SRCS))

all: $(appname)

$(appname): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(appname) $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(appname) $(OBJS)

dist-clean: clean
	$(RM) *~ .depend

include .depend
