#!/bin/bash

. ./Configuration.sh

# Prepare data

for (( i=0; i<=840; i+=60 ))
do
	printf -v j "%03d" $i
	./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild_$j.ppm -crop 1080x1080+$i+0
done

./FfmpegSample -if "./2k_wild_*.ppm" -o ffmpeg.avi -frameRepeat 8 -MaxWidth 1080 -MaxHeight 1080 -info -log ffmpeg.log
./FfmpegSample -if "./2k_wild_*.ppm" -o ffmpeg_async.avi -frameRepeat 8 -MaxWidth 1080 -MaxHeight 1080 -async -info
./FfmpegSample -i ffmpeg.avi -o "./decoded_*.ppm"
