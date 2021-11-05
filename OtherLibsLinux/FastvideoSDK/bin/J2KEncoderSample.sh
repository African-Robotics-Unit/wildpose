#!/bin/bash

. ./Configuration.sh
# Prepare data
# Create 12 bits image
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.12.ppm -bits 12 -shift 4 
# Create 10 bits image (max value is 1023) in 12 bits ppm (max value 4095)
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.10.12.ppm -bits 12 -shift 2 

# Lossless Single Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossless.jp2 -a rev -l 7 -c 32 

# Lossy Single Mode
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.jp2 -a irrev -l 7 -c 32 -q 80
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.444.jp2 -a irrev -l 7 -c 32 -q 80 -s 444
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.422.jp2 -a irrev -l 7 -c 32 -q 80 -s 422
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.420.jp2 -a irrev -l 7 -c 32 -q 80 -s 420

# Lossy Single Mode for 10 and 12 bits images
# Compress 12 bits image in 12 bits jp2
./J2KEncoderSample -i 2k_wild.12.ppm -o 2k_wild.12.lossy.jp2 -a irrev -l 7 -c 32 -q 80 -s 444
# Compress 12 bits image in 10 bits jp2. All pixels are devided by 4 to fit in 10 bits range
./J2KEncoderSample -i 2k_wild.12.ppm -o 2k_wild.12.lossy.jp2 -a irrev -l 7 -c 32 -q 80 -s 444 -outputBitdepth 10

# Compress 10 bits image (max value is 1023) in 12 bits range (max value 4095)
# Compress as 12 bits image.
./J2KEncoderSample -i 2k_wild.10.12.ppm -o 2k_wild.12.10.lossy.jp2 -a irrev -l 7 -c 32 -q 80 -s 444
# Overwrite range of source image to 10 bits and store image as 10 bits. There is no devide by 4.
./J2KEncoderSample -i 2k_wild.10.12.ppm -o 2k_wild.10.lossy.jp2 -a irrev -l 7 -c 32 -q 80 -outputBitdepth 10 -s 444 -overwriteSourceBitdepth 10


# Other Option
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.noMCT.jp2 -a irrev -l 7 -c 32 -q 90 -noMCT
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.noHeader.jp2 -a irrev -l 7 -c 32 -q 90 -noHeader
./J2KEncoderSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.lossy.tiled.jp2 -a irrev -l 7 -c 32 -q 90 -tileWidth 128 -tileHeight 128
