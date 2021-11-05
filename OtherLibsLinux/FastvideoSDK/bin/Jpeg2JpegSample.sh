#!/bin/bash

. ./Configuration.sh

./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.default.jpg -outputWidth 1023 -q 90 -r 4 -s 444 -info -log Jpeg2Jpeg.log
./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.sharp.after.jpg -sharp_after 0.95 -outputWidth 1025 -q 90 -r 4 -s 444 -info
./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.sharp.before.jpg -sharp_before 0.95 -outputWidth 1022 -q 90 -r 4 -s 444 -info -maxWidth 4096 -maxHeight 4096
./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.sharp.after.before.jpg -sharp_before 0.95 -sharp_after 0.95 -outputWidth 1022 -q 90 -r 4 -s 444 -info -maxWidth 4096 -maxHeight 4096
./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.sharp.after.before.3000.jpg -sharp_before 0.95 -sharp_after 0.95 -outputWidth 3000 -q 90 -r 4 -s 444 -info -maxWidth 4096 -maxHeight 4096
./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.crop.jpg -outputWidth 1023 -crop 1600x1000+12+10 -q 90 -r 4 -s 444 -info

./Jpeg2JpegSample -i $DATA_SET/Images/2k_wild.ri1.jpg -o 2k_wild.ri1.crop.mt.jpg -outputWidth 1023 -crop 1600x1000+12+10 -q 90 -r 4 -s 444 -repeat 64 -async -thread 2 -threadR 2 -threadW 2 -b 2 -info
