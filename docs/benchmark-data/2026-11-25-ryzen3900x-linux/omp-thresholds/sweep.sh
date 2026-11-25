#!/bin/bash
cd "$(dirname "$0")"
cat /proc/loadavg > load-start.txt
for r in 1 2; do
  : > sweep-r$r.txt
  ./thr mixed15 1,2,4,8,12 6 17 >> sweep-r$r.txt
  ./thr mixed3 1,2,4,8,12 8 19 >> sweep-r$r.txt
  ./thr real 1,2,4,8,12 12 22 >> sweep-r$r.txt
  ./thr dct 1,2,4,8,12 12 22 >> sweep-r$r.txt
  ./thr bluestein 1,2,4,8,12 9 20 >> sweep-r$r.txt
  ./thr pow2 1,2,4,8,12 13 22 >> sweep-r$r.txt
  cat /proc/loadavg >> load-start.txt
done
echo done > finished
