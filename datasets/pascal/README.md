# PASCAL VOC and SBD

PASCAL VOC is a standard recognition dataset and benchmark with detection and semantic segmentation challenges.
The semantic segmentation challenge annotates 20 object classes and background.
The Semantic Boundary Dataset (SBD) is a further annotation of the PASCAL VOC data that provides more semantic segmentation and instance segmentation masks.

SBD and VOC 2012 both diverged after VOC 2011. SBD is a further annotated version of PASCAL 2011. And, PASCAL VOC 2012 is basically the previous 2008-2011 data with more segmentation annotations over existing images. Thus, the train/val/test splits of PASCAL VOC 2012 segmentation challenge and SBD diverges. Most notably VOC 2011 validation images intersects with SBD train.
Care must be taken for proper evaluation by excluding images from the train or val splits.

We train on the 8,498 images of SBD train: 
    benchmark.tgz -> benchmark_RELEASE

We validate on the non-intersecting set defined in the included `seg11valid.txt`. Although we are using the official PASCAL VOC 2012 dataset, we are only taking a subset of the PASCAL VOC 2011 validation images that are not in the SBD training set.
    VOCtrainval_11-May-2012.tar -> VOC2012

PASCAL VOC has a private test set and [leaderboard for semantic segmentation](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6).

See the dataset sites for download:
- PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- SBD: see [homepage](http://home.bharathh.info/home/sbd) or [direct download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

PASCAL classes:
0:  background
1:  aeroplane
2:  bicycle
3:  bird
4:  boat
5:  bottle
6:  bus
7:  car
8:  cat
9:  chair
10: cow
11: diningtable
12: dog
13: horse
14: motorbike
15: person
16: pottedplant
17: sheep
18: sofa
19: train
20: tvmonitor

and 255 is the ignore label that marks pixels excluded from learning and
evaluation by the PASCAL VOC ground truth.

