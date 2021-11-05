#!/bin/bash

. ./Configuration.sh

# Prepare data
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.rev.jp2 -a rev -l 7 -c 32 

# Single Mode
./J2KDecoderSample -i 2k_wild.REV.jp2 -o 2k_wild.rev.ppm -repeat 100 -discard 
# Batch Mode
./J2KDecoderSample -i 2k_wild.REV.jp2 -o 2k_wild.rev.ppm -repeat 100 -b 4 -discard 
# Multithread-Batch Mode
./J2KDecoderSample -i 2k_wild.REV.jp2 -o 2k_wild.rev.ppm -repeat 100 -b 4 -async -thread 2 -threadR 2 -threadW 2 -info -discard 

# Single Mode
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.IRREV.ppm -repeat 100 -discard 
# Batch Mode
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.IRREV.ppm -repeat 100 -b 4 -discard 
# Multithread-Batch Mode
./J2KDecoderSample -i $DATA_SET/Images/2k_wild.IRREV.jp2 -o 2k_wild.IRREV.ppm -repeat 100 -b 4 -async -thread 2 -threadR 2 -threadW 2 -info -discard
