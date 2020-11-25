# Regular use
CFLAGS=-O3
ARCH := amd64
#machine architecture flags
ifeq ($(ARCH),arm)
	#make ARCH=arm
	MFLAGS=-march=armv8-a+fp+simd+crypto+crc
else
	MFLAGS=-march=native
endif


#run "make" for default build

ifneq ($(OS),Windows_NT)
	OS = $(shell uname)
 	ifeq "$(OS)" "Darwin"
 
		#MacOS links g++ to clang++, for gcc install via homebrew and replace g++ with /usr/local/bin/gcc-9		
	endif
endif



all:
	g++ $(CFLAGS) -o tst main.cpp $(MFLAGS)