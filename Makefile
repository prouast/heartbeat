# Makefile for heartbeat
appname := Heartbeat
INPUT ?=

CXX := g++
RM := rm -f
CXXFLAGS := -Wall -g -std=c++11 -I/usr/local/include/opencv4 -I/usr/include/opencv4 -I.
LDFLAGS := -g
LDLIBS := -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_videoio

# Find all source files, excluding object files
SRCS := $(shell find . -name "*.cpp" ! -path "*/.*" ! -name "*~")
OBJS = $(subst .cpp,.o,$(SRCS))

all: $(appname)

.PHONY: run-deep run-deep-live run-deep-file reset
run-deep: run-deep-live

run-deep-live: $(appname)
	./$(appname) -facedet deep -gui true

run-deep-file: $(appname)
	@if [ -z "$(INPUT)" ]; then \
		echo "Missing INPUT. Usage: make run-deep-file INPUT=/absolute/path/video.mp4"; \
		exit 1; \
	fi
	@if [ ! -f "$(INPUT)" ]; then \
		echo "Input file not found: $(INPUT)"; \
		echo "Current directory: $$(pwd)"; \
		echo "Tip: use an absolute path, e.g. INPUT=/Users/aarush/Downloads/video.mp4"; \
		exit 1; \
	fi
	./$(appname) -i "$(INPUT)" -facedet deep -gui true

reset:
	$(RM) ./.depend
	$(RM) $(OBJS) $(appname)
	$(MAKE) depend
	$(MAKE) $(appname)

$(appname): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(appname) $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > ./.depend

clean:
	$(RM) $(appname) $(OBJS)

dist-clean: clean
	$(RM) *~ .depend

-include .depend
