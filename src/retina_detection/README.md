# RetinaNet Object Detection

This is a standalone repository for training a RetinaNet object detector.

# Usage

Here we'll describe what you need to train a model for yourself.

## Dataset

A dataset of tires was manually collected from Google Images. With a set of images,
head over to [MakeSense.ai](https://www.makesense.ai/) and label your image set. You
can use any labelling service which results in a COCO style dataset, but there are
some utility scripts than can prepare an output from MakeSense.ai. 

Once the data is labelled, export the annotations as YOLO txt files. Next use the
`src/retina_detection_train/train_utils/yolo_to_coco.py` to turn the YOLO style datset
into a COCO-like object detection dataset. Make sure to adjust the relavant options at
the top of the script, mainly where your labels are and where you want to save the data.

At this point you now have a COCO-like dataset that can be used for training. This
dataset will only have an `annotation.json` file whereas a real COCO dataset would have
annotation files for all the different data splits.



Training runs are configured by a `.yaml` file inside `train/configs`. Right now there
is just one config for a retinanet18 model. 

