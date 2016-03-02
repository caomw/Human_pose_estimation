CC = gcc
CXX = g++ -O3 -fopenmp
OPT = -std=c++0x
BOOST_ROOT = /usr/local
INC = -I$(BOOST_ROOT)/include
LDFLAGS = -L$(BOOST_ROOT)/lib
LDFLAGS += -lboost_system -lboost_filesystem

all: main.cpp
	$(CXX) $(OPT) -o estPose main.cpp $(INC) $(LDFLAGS) `pkg-config --cflags --libs opencv`
