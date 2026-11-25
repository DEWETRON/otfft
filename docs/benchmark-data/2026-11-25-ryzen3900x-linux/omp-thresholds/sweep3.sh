#!/bin/bash
cd "$(dirname "$0")"
cat /proc/loadavg > load3-start.txt
: > sweep3.txt
./thr.v1 mixed15 1,2,4,12 11 17 >> sweep3.txt
./thr.v1 mixed3 1,2,4,12 13 19 >> sweep3.txt
./thr.v1 real 1,2,4,12 14 22 >> sweep3.txt
./thr.v1 dct 1,2,4,12 14 22 >> sweep3.txt
cat /proc/loadavg >> load3-start.txt
echo done > finished3
