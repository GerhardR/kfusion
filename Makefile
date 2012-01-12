CXX = nvcc
CC = nvcc
CPPFLAGS=-I../include -I/usr/X11/include -I/usr/loca/cuda/include -I/opt/local/include
CXXFLAGS=-m64 -O3 -use_fast_math
LDFLAGS=-m64 -L../lib -L/usr/X11/lib -lGL -lcvd -lfreenect

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $^

all: kinect test

test: kfusion.o helpers.o test.o

kinect: kfusion.o helpers.o kinect.o

clean:
	rm *.o test kinect

