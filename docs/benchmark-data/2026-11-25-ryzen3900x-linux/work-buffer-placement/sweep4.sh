#!/bin/bash
cd "$(dirname "$0")"
cat /proc/loadavg > load4-start.txt
: > sweep4.txt
./thr.fix mixed15 1 8 17 >> sweep4.txt
./thr.fix mixed3 1 10 19 >> sweep4.txt
./thr.fix real 1 12 20 >> sweep4.txt
./thr.fix dct 1 12 20 >> sweep4.txt
./thr.fix bluestein 1 11 18 >> sweep4.txt
./thr.fix pow2 1 13 20 >> sweep4.txt
cat /proc/loadavg >> load4-start.txt
echo done > finished4
