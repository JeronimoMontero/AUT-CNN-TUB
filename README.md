# AUT-CNN-TUB
Deep-learning object identification for robot-based assembly

## Data Excess

Data Excess will be granted via AWS S3, therefore  you need an VPN connection.

+ Raw Data:

    https://s3.us-east-2.amazonaws.com/imagesforcnn/Datensatz.zip

+ Subset of centered and well sized Images:

    https://s3.us-east-2.amazonaws.com/imagesforcnn/Centered.zip

+ Tensorflow Dataset of the centered Images:

    https://s3.us-east-2.amazonaws.com/imagesforcnn/Dataset.zip

+ Trained Tensorflow Models which have been proven:

    https://s3.us-east-2.amazonaws.com/imagesforcnn/model_???.zip

## Prepossessing

### Split into Subimages for each part

This would be necessary if we don't get an image fore each part.

### Centering Image
+ http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

### Work with limited Data:
+ https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

### Generate Tensorflow Dataset:

## Model

## Training

## Predictor

## Maybe later
Function which gets several Images from the same Part in different  angels, to calculate the most likely Name for the Part.

