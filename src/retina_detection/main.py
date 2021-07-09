import pathlib
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import torch
import numpy as np

from model import detector


def normalize(
    img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
) -> np.ndarray:

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator

    return img


'''mk
save_path = "data/result/"
testdata_path = "data/"
num_data = 10
# image = cv2.imread(
    # "/home/alex/Desktop/projects/minimal-object-detector/src/train/data/images/2020-Toyota-86-GT-TRD-Wheels.jpg"
# )
for i in range(num_data):
    filename= testdata_path+str(i)+".jpg"
    print("test filename", filename)
    image_flip = cv2.imread(filename)
    image = cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB)
    # image = cv2.imread(filename)

'''
# save_path = pathlib.Path("/tmp/results/")
# save_path = "data/result/"
save_path = pathlib.Path("/home/mk/project/retinanet_ros/src/retina_detection/data/result/")
testdata_path = pathlib.Path("/home/mk/project/retinanet_ros/src/retina_detection/data/")
model = detector.Detector(timestamp="2021-07-01T22.24.44")
model.eval() 
for img in testdata_path.glob("*"):
    if img.is_dir()==True:
        continue
    print("test filename", img)
    image_flip = cv2.imread(str(img))
    image = cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB)
    image_before= image
    image_before= cv2.cvtColor(image_before, cv2.COLOR_RGB2BGR)
    origin_shape = image.shape
    o_height, o_width= image.shape[:2]
    print("o_width", o_width)
    print("o_height", o_height)
    target_size=512
    x_scale = target_size/o_width 
    y_scale = target_size/o_height 
    # print("x_scale", x_scale)
    # print("y_scale", y_scale)
    image_ori = cv2.resize(image, (512, 512))
    image_flip = cv2.resize(image_flip, (512, 512))
    image = normalize(image_ori)


    with torch.no_grad():
        image = torch.Tensor(image)
        if torch.cuda.is_available():
            image = image.cuda()
        boxes = model.get_boxes(image.permute(2, 0, 1).unsqueeze(0))

    for box in boxes[0]:
        confidence = float(box.confidence)
        box = (box.box * torch.Tensor([512] * 4)).int().tolist()

        if confidence > 0.05:
            print(confidence)
            box[0]=int(np.around(box[0]*1/x_scale))
            box[2]=int(np.around(box[2]*1/x_scale))
            box[1]=int(np.around(box[1]*1/y_scale))
            box[3]=int(np.around(box[3]*1/y_scale))
            print("box[0]", box[0])
            print("box[1]", box[1])
            print("box[2]", box[2])
            print("box[3]", box[3])
            cv2.rectangle(image_before, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)

    # image_before= cv2.cvtColor(image_before, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path / img.name), image_before)
    print("sve_path", save_path)
