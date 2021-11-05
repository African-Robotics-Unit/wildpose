#!/bin/bash

. ./Configuration.sh
#JPEG encoder (Single thread)
./JpegSample -i $DATA_SET/Images/2k_wild.ppm -o ./2k_wild.jpg -repeat 100 
#JPEG encoder (Multi thread)
./JpegSample -i $DATA_SET/Images/2k_wild.ppm -o ./2k_wild.mt.jpg -repeat 200 -async -thread 2 -threadR 2 -threadW 1 -b 2 -info

#JPEG decoder (Single thread)
./JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o ./2k_wild.ppm -repeat 100
#JPEG decoder (Multi thread)
./JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o ./2k_wild.mt.ppm -repeat 200 -async -thread 2 -threadR 2 -threadW 1 -b 2 -info

