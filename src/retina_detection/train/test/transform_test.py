import albumentations as A
import cv2

bboxes = [[1668,1230,560,785]]
category_ids=[1]

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=[0.2, 0.5]),
    # A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.3,p=0.8)],
    # A.Perspective(0.9,keep_size=True)],
    # A.RandomSizedBBoxSafeCrop(1500, 1000, erosion_rate=0.2)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    # A.Perspective(0.3,keep_size=False)
)

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("src/retina_detection/train/test/image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
transformed_image = transformed["image"]

outputname ="src/retina_detection/train/test/transform_image.jpg"
cv2.imwrite(outputname, transformed_image)
