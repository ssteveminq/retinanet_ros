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


## Training

Training runs are configured by a `.yaml` file inside
`src/retina_detection_train/train/configs`. Right now there is just one config for a
retinanet18 model. 

In the config, find the `data.data_path` key and change the path to your dataset.

To start a training job, run:

```
PYTHONPATH=. train/train.py \
    --config train/configs/retinanet18.yaml
```

Training will automatically use all the available GPUs, but sometimes this is
problematic if you have different cards. To only traing on one or some set of
gpus, use:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 train/train.py \
    --config train/configs/retinanet18.yaml
```

For two GPUs, `CUDA_VISIBLE_DEVICES=0,1`. And `CUDA_VISIBLE_DEVICES=-1` will run on a
cpu.


This will save model checkpoints to `~/runs/tire-detector` as specified at the top of
`train.py`. Feel free to change the save directory to something that makes more sense
for you.


## Contents

* `model/`: a basic RetinaNet implementation.
* `third_party/`: external code for object detection. Mainly from
  [`detectron2`](https://github.com/facebookresearch/detectron2).
* `train/`: the bulk of relevant traing code.
    * `configs/`: where the .yaml training configuration files are held.
    * `train_utils/`: some scripts either directly used during training or help create
      a sucessful training run.
      * `logger.py`: simple script used to log output during training.
      * `utils.py`: utility functions to aid training.
      * `yolo_to_coco.py`: convert YOLO labels to COCO dataset.
    * `augmentations.py`: the training and validation augmentations used during training.
    * `collate.py`: process per-batch training data for training.
    * `dataset.py`: training dataset built around a COCO dataset.
    * `train.py`: the main training script.
      
