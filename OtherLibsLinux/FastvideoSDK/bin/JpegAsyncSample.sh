#!/bin/bash

. ./Configuration.sh

./JpegAsyncSample -i $DATA_SET/Images/2k_wild.ppm -o 2k_wild.444.jpg -q 95 -s 444 -info -log jpeg_async.log
