#!/usr/bin/env bash

EXPERIMENT=$1

set -e

ls -1 $EXPERIMENT/*/metrics.json | \
parallel 'echo {} $(jq .validation_accuracy_avg {})' | \
sort | \
sed -Ee 's/^[^/]*\/+//' | \
sed -e "s/\/metrics.json//" | \
sed -Ee 's/\.json_([0-9][0-9]-){6}r[0-9]+//' | \
csvtool -t " " -u TAB transpose -
