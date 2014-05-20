# When editing this Makefile, please make sure all variables apart from
# the KF_* and ALL_* variables stay overridable from outside.
# KF_* variables *may* be set from outside (like KF_DRIVER).

CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

# Can be either 'libfreenect' or 'openni2'.
# If not given, it's 'libfreenect' by default.
KF_DRIVER ?= libfreenect

# You can set the following path variables for this Makefile:
#   CUDA_INCLUDE_PATH
#   OPENNI2_INCLUDE_PATH
#   LIBFREENECT_INCLUDE_PATH


# Default flags
KF_CPPFLAGS = -Ithirdparty
KF_CXXFLAGS = -g -m64 -O3
KF_NVCCFLAGS = -g -m64 -O3 -use_fast_math
KF_LDFLAGS = -g -m64 -lpthread -Xlinker -lcudart

ifdef CUDA_INCLUDE_PATH
  KF_CPPFLAGS += -I$(CUDA_INCLUDE_PATH)
endif

# Driver dependent flags

# For using libfreenect
ifeq ($(KF_DRIVER),libfreenect)
  KF_CPPFLAGS += -DLIBFREENECT_INTERFACE
  ifdef LIBFREENECT_INCLUDE_PATH
    KF_CPPFLAGS += -I$(LIBFREENECT_INCLUDE_PATH)
  endif
  KF_LDFLAGS += -lfreenect
# For using OpenNI2
else ifeq ($(KF_DRIVER),openni2)
  KF_CPPFLAGS += -DOPENNI2_INTERFACE
  ifdef OPENNI2_INCLUDE_PATH
    KF_CPPFLAGS += -I$(OPENNI2_INCLUDE_PATH)
  endif
  KF_LDFLAGS += -lOpenNI2
else
  $(error KF_DRIVER is not set to a possible driver)
endif

# OS dependent flags

KF_OS := $(shell uname)
ifeq ($(KF_OS),Darwin)
  KF_LDFLAGS += -framework,OpenGL,-framework,GLUT
else
  KF_LDFLAGS += -lGL -lglut
endif


# Concatenate our and user flags
ALL_CPPFLAGS := $(KF_CPPFLAGS) $(CPPFLAGS)
ALL_CXXFLAGS := $(KF_CXXFLAGS) $(CXXFLAGS)
ALL_NVCCFLAGS := $(KF_NVCCFLAGS) $(NVCCFLAGS)
ALL_LDFLAGS := $(KF_LDFLAGS) $(LDFLAGS)


# Default target
.PHONY: all
all: kinect test


# .cpp files
%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) $(ALL_CPPFLAGS) -c -o $@ $<

# .cu files
%.cu_o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_CPPFLAGS) -c -o $@ $<


# Executables

kinect: kinect.cpp kfusion.cu_o helpers.cu_o interface.o
	$(CXX) $(ALL_CPPFLAGS) $(ALL_CXXFLAGS) $^ -o $@ $(ALL_LDFLAGS)

test: test.cpp kfusion.cu_o helpers.cu_o
	$(CXX) $(ALL_CPPFLAGS) $(ALL_CXXFLAGS) $^ -o $@ $(ALL_LDFLAGS)


# Cleanup
.PHONY: clean
clean:
	rm -f *.cu_o *.o test kinect
