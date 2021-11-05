#!/bin/bash

. ./Configuration.sh

# Prepare data
./ImageConverterSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.1920x1080.RGGB.pgm -pattern RGGB

for (( i=0; i<=840; i+=60 ))
do
	printf -v j "%03d" $i
	./ImageConverterSample -i 2k_wild.1920x1080.RGGB.pgm -o 2k_wild_$j.pgm -crop 1080x1080+$i+0
done

./ImageConverterSample -i 2k_wild_000.pgm -o 2k_wild.1080x1080.MatrixA.1.1.pfm -matrix -pixel 1.1
./ImageConverterSample -i 2k_wild_000.pgm -o 2k_wild.1080x1080.MatrixB.30.pgm -matrix -pixel 30

./CameraSample -if "./2k_wild_*.pgm" -o sample_1.avi -maxWidth 1080 -maxHeight 1080 -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -lut $DATA_SET/LUT/lut.txt -frameRepeat 24 -log camera.log
./CameraSample -if "./2k_wild_*.pgm" -o sample_2.avi -maxWidth 1080 -maxHeight 1080 -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -lut $DATA_SET/LUT/InvertColor.8bit.txt -frameRepeat 24
./CameraSample -if "./2k_wild_*.pgm" -o sample_3.avi -maxWidth 1080 -maxHeight 1080 -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -matrixA 2k_wild.1080x1080.MatrixA.1.1.pfm -matrixB 2k_wild.1080x1080.MatrixB.30.pgm -lut $DATA_SET/LUT/InvertColor.8bit.txt -frameRepeat 24
