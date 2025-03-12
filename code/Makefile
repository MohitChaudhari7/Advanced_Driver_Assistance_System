INCLUDE_DIRS = -I/usr/local/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -lopencv_cudaarithm -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_imgproc -lopencv_highgui -lpthread

HFILES= 
CFILES= lane_haar.cpp cuda_lane_haar.cpp yolo.cpp cudnn_yolo.cpp cudnn_yolo_lane.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	lane_haar cuda_lane_haar yolo cudnn_yolo cudnn_yolo_lane

clean:
	-rm -f *.o *.d
	-rm -f lane_haar cuda_lane_haar yolo cudnn_yolo cudnn_yolo_lane

lane_haar: lane_haar.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

cuda_lane_haar: cuda_lane_haar.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

yolo: yolo.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

cudnn_yolo: cudnn_yolo.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

cudnn_yolo_lane: cudnn_yolo_lane.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<
