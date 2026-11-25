#!/bin/bash
cd "$(dirname "$0")"
cat /proc/loadavg > scan2-load.txt
./scanoff.sh complex scan-heldout.txt $(cat heldout.txt)
cat /proc/loadavg >> scan2-load.txt
echo done > scan2-finished
