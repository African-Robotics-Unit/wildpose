#!/bin/bash

. ./Configuration.sh

# Lossless Single Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.jp2 -a rev -l 7 -c 32 -repeat 100 -discard -log j2k.log
# Lossless Batch Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.batch.jp2 -a rev -l 7 -c 32 -b 4 -repeat 200 -discard
# Lossless Multithread-Batch Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.multi.jp2 -a rev -l 7 -c 32 -b 4 -repeat 200 -async -thread 2 -threadR 2 -threadW 1 -b 2 -info -discard

# Lossy Single Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.jp2 -a irrev -l 7 -c 32 -q 80 -repeat 100 -discard
# Lossy Batch Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.batch.jp2 -a irrev -l 7 -c 32 -q 80 -b 8 -repeat 200 -discard
# Lossy Multithread-Batch Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.multi.jp2 -a irrev -l 7 -c 32 -q 80 -b 8 -repeat 200 -async -thread 2 -threadR 2 -threadW 1 -b 2 -info -discard
