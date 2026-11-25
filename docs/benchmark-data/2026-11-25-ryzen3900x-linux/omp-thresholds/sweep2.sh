#!/bin/bash
cd "$(dirname "$0")"
cat /proc/loadavg > load2-start.txt
: > sweep2.txt
./thr mixed15 1,2,4,12 8 17 >> sweep2.txt
./thr mixed3 1,2,4,12 10 19 >> sweep2.txt
./thr real 1,2,4,12 12 22 >> sweep2.txt
./thr dct 1,2,4,12 12 22 >> sweep2.txt
./thr bluestein 1,2,4,12 11 20 >> sweep2.txt
./thr pow2 1,2,4,12 13 22 >> sweep2.txt
cat /proc/loadavg >> load2-start.txt
echo done > finished2
