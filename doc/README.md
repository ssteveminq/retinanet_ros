# Minimal Object Detector


Some code to run an object detector in PyTorch with python. This model is not trained at
all so it won't find anything useful.

## Example Setup
In a virtual environment:

```
python3 -m venv ~/.envs/minimal-od
source ~/.envs/minimal-od/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```


## Example

[Here](https://user-images.githubusercontent.com/31543169/115086127-4e10a800-9ed1-11eb-8d0c-919587538381.jpg)
is an example image.


Update the image path in the script, then run the main script like so:

```
PYTHONPATH=. python src/main.py
```


## Python3 vs Python2

I haven't tested this in python2. We would have to try building PyTorch for python2. We could also look at using C++.
