# Minimal Object Detector


Some code to run an object detector in PyTorch with python. This model is not trained at
all so it won't find anything useful.

## Example

[Here](https://user-images.githubusercontent.com/31543169/115086127-4e10a800-9ed1-11eb-8d0c-919587538381.jpg)
is an example image.


Update the image path in the script, then run the main script like so:

```
PYTHONPATH=. python src/main.py
```

## Models

The object detector here is a [RetinaNet](https://arxiv.org/abs/1708.02002) model.

The backbone of the model is a Resnet variant. We opt to use the
Retinanet-18 variant instead of the heavier options because we are
only training the model with one class. The other models are still
available.

## Training

Check out the [training instructions](/src/train/README.md).
