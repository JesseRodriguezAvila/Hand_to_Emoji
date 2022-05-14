# Emoji-Hand
Emoji-Hand : A Computer Vision Application To Predict Hand Emojis From Hand-poses


Our project consist of 5 steps:
1. collecting our own data
2. labeling (Image annotation)
3. create label_map.pbtxt
4. create train.record and test.record
5. fine-tune tensorflowws pre-trained model SSD MobileNetV2
6. Train model: SSD MobileNetV2
7. real-time detections of our hand gestures

## collecting our own data



## Step 4.) Creating train.record and test.record

Tensorflow Object Detection API provided a script, generate_tfrecord.py, that will generate train.record and test.record

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records

To run this scipt, we need to run generate_tfrecord.py with the following parameters:

python [dir of generate_tfrecords.py] -x [dir of train or test images] -l [dir of label_map.pbtxt] -o [dir of where to save train.record or test.record]

## BEFORE MOVINF ON, WE NEED TO DOWNLOAD Tensorflow models FROM Tensorflow model zoo 
https://github.com/tensorflow/models

http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

we then need to copy the pipeline.congif file from ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz, then create a new folder in the models folder. in our case we named it 'ssd_mobilenet'. we need to paste that pipeline.config file inside the newly created file

### Step 5.) fine-tune tensorflowws pre-trained model SSD MobileNetV2 using transfer learning


### Step 6.) Train model: SSD MobileNetV2

To train tensorflows model we need to run the script model_main_tf2.py provided by tensorflow 
for example:

!python /content/drive/MyDrive/FinalProject/Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=/content/drive/MyDrive/FinalProject/Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=/content/drive/MyDrive/FinalProject/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000

