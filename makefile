# Regular use
CFLAGS=-s -O3

#run "make" for default build

ifneq ($(OS),Windows_NT)
	OS = $(shell uname)
 	ifeq "$(OS)" "Darwin"
 
		#MacOS links g++ to clang++, for gcc install via homebrew and replace g++ with /usr/local/bin/gcc-9		
	endif
endif
all:
	g++ $(CFLAGS) -o tst main.cpp -march=native