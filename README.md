# Emoji-Hand
Emoji-Hand : A Computer Vision Application To Predict Hand Emojis From Hand-poses


Our project consist of 5 steps:
1. collecting our own data
2. labeling (Image annotation)
3. create label_map.pbtxt
4. create train.record and test.record
5. fine-tune tensorflowws pre-trained model SSD MobileNetV2
6. real-time detections of our hand gestures

### collecting our own data



### create train.record and test.record

Tensorflow Object Detection API provided a script, generate_tfrecord.py, that will generate train.record and test.record

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records
