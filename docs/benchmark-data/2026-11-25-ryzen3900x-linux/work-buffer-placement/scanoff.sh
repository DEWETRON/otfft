#!/bin/bash
# usage: scanoff.sh KIND OUTFILE N...
kind=$1; out=$2; shift 2
: > $out
for off in $(seq 0 64 4032); do
  echo "# off $off" >> $out
  OTFFT_WORK_OFFSET=$off ./scanlib $kind "$@" >> $out
done
