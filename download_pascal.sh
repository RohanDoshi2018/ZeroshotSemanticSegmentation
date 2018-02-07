#!/bin/bash
DIR=$(cat "data_dir.txt")
mkdir -p $DIR/pascal

cp "pascal/README.md" $DIR/pascal/README.md
cp "pascal/seg11valid.txt" $DIR/pascal/seg11valid.txt

cd $DIR/pascal

if [ ! -e benchmark_RELEASE ]; then
  wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz -O benchmark.tar
  tar -xvf benchmark.tar -C benchmark_RELEASE --strip-components=1
fi

if [ ! -e VOC2012 ]; then
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_11-May-2012.tar -C VOC2012 --strip-components=1
fi