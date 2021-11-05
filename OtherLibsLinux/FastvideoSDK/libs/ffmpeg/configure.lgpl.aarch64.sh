#!/bin/bash

./configure --cc="gcc" --enable-asm --disable-doc --disable-ffplay --disable-ffprobe --enable-ffmpeg --enable-shared --disable-static --disable-bzlib --disable-libopenjpeg --disable-iconv --disable-avdevice  --disable-swscale --disable-postproc --disable-avfilter  --prefix=bin --arch=aarc64

make
make install