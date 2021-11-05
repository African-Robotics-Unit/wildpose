/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "timing.hpp"

#ifndef __GNUC__
#include <windows.h>

struct hostTimerStruct{
private:
    LARGE_INTEGER _freq, _start, _stop;

public:
    hostTimerStruct(){
        _freq.QuadPart = 0;
        _start.QuadPart = 0;
        _stop.QuadPart = 0;
        QueryPerformanceFrequency(&_freq);
    }

    void hostTimerStart(){
        QueryPerformanceCounter(&_start);
    }

    double hostTimerEnd(){
        QueryPerformanceCounter(&_stop);
        return (double(_stop.QuadPart) - double(_start.QuadPart)) / double(_freq.QuadPart);
    }

    ~hostTimerStruct(){}
};

#else
#include <sys/time.h>
#include <cstdio>

struct hostTimerStruct
{
  double start;
public:
  hostTimerStruct() : start(0.) {};
  void hostTimerStart()
  {
    struct timeval _start;
    gettimeofday(&_start,NULL);
    start = _start.tv_sec*1. +_start.tv_usec*1.e-6;
  }
  double hostTimerEnd()
  {
    struct timeval _stop;
    gettimeofday(&_stop,NULL);
    double stop = _stop.tv_sec*1. +_stop.tv_usec*1.e-6;
    return stop-start;
  }
};
#endif


hostTimer_t hostTimerCreate(){
    return new hostTimerStruct;
}

void hostTimerStart(hostTimer_t timer){
    timer->hostTimerStart();
}

double hostTimerEnd(hostTimer_t timer){
    return timer->hostTimerEnd();
}

void hostTimerDestroy(hostTimer_t timer){
    delete timer;
}


void hostTimerStart(hostTimer_t timer, bool predicate)
{
    if (predicate)
        hostTimerStart(timer);
}

double hostTimerEnd(hostTimer_t timer, bool predicate)
{
    if (predicate)
        return hostTimerEnd(timer);

    return 0.0;
}
