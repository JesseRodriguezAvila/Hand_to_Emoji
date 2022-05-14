# Emoji-Hand
Emoji-Hand : A Computer Vision Application To Predict Hand Emojis From Hand-poses

Our code was written in jupyter notebook (.ipynb)

The project consist of 5 steps:
1. collecting our own data
2. labeling (Image annotation)
3. split our custom data into training and testing sets
4. create label_map.pbtxt
5. create train.record and test.record
6. fine-tune tensorflowws pre-trained model SSD MobileNetV2
7. Train model: SSD MobileNetV2
8. real-time detections of our hand gestures

## STEP 1.) collecting our own custom dataset

For the collection of our data we used Python3 and openCV.

we created a jupyter notbook file called dataset_capture.ipynb

In this file a function data_collect(label, num_images) can be envoked which will allows us to take n number of image for a handpose. This process is then repeated for a number of desired hand gestures / poses

## STEP 2.) We used the open-source labelImg.py package to manually apply to all of our collected images an image annotation




## STEP 3.) split our custom data into training and testing sets

once image annotation is completed, we splitted our data.

for this we a script under filename splt_data.ipynb

we borrowed code (def xml_to_csv) from generate_tfrecord.py to get the names of all different images under a folder. this function will grab the name of the images and class (label) from their respective xml files.

secondly, we will know the names of all images, we then seperated them by hand gesture / class. 

once separated by class, for every class we randomized them and seleted 26% for testing (head of list). this was done by getting the length of all images within one gesture type, multiplied times n percent. This will determine how many will go to testing. the remaining will go to training (tail of list).


for example -->
[

        {"label": ToRight, "train": [imagefilenames of 11], "test": [imagefilenames of 4 images]}, 
        
        {"label": Ok, "train": [imagefilenames of 11], "test": [imagefilenames of 4 images]},
        
        ...  
]

we now know which images we want to testing and which for training, from an image filename, we get the name and remove .jpg and replace it with .xml

this will leave us with tew files

someimage_111.jpg
someimage_111.xml

then we used shutil.copy both jpg and xml into thier respective train or test folder



# From step 4 - 7 we worked on Google collab because of our limited computing power. These steps are done under the jupyter notebook file transferlearning_pretrainModel.ipynb

## Step 4.) Creating label_map.pbtxt

we have a list containing the labels for all our hand gestures.

First, we create a dict list for all our hand gestures. in this list every dict contains id and name.

Second, after creating dict list, we then envoke a function called creat_label_map_file which will create our label_map.pbtxt. this function requires two parameters, a path where to save the generated file and the dict list.


## Step 5.) Creating train.record and test.record

Tensorflow Object Detection API provided a script, generate_tfrecord.py, that will generate train.record and test.record

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records

To run this scipt, we need to run generate_tfrecord.py with the following parameters:

python [dir of generate_tfrecords.py] -x [dir of train or test images] -l [dir of label_map.pbtxt] -o [dir of where to save train.record or test.record]

## BEFORE MOVINF ON, WE NEED TO DOWNLOAD Tensorflow models FROM Tensorflow model zoo 
https://github.com/tensorflow/models

http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

we then need to copy the pipeline.congif file from ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz, then create a new folder inside models folder. In our case we named it 'ssd_mobilenet'. we need to paste that pipeline.config file inside the newly created file



## Step 6.) fine-tune tensorflowws pre-trained model SSD MobileNetV2 using transfer learning

we need to update our pipeline.config inside the models/ssd_mobilenet folder with the following parameters

because we have 11 hand gestures

1.) pipeline_config.model.ssd.num_classes = 11

batch size of 4

2.) pipeline_config.train_config.batch_size = 4

path to pre-trained model

3.) pipeline_config.train_config.fine_tune_checkpoint = (pathto_pre-trainedmodel)+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'

detection type

4.) pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

generated label_map

5.) pipeline_config.train_input_reader.label_map_path= (pathto_label_map) + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].label_map_path = (pathto_label_map) + '/label_map.pbtxt'

generated train.record

6.) pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [(pathto_train.record) + '/train.record']

generated test.record

7.) pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [(pathto_test.record) + '/test.record']



## Step 7.) Train model: SSD MobileNetV2

To train tensorflows model we need to run the script model_main_tf2.py provided by tensorflow 
for example:

!python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=/Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000

we need the file provided by tensorflow Api: model_main_tf2.py

we need directory for checkpoint to save

we need our configured pipeline.config

we also need to determine which num_train_step will give us the best results

# For step 8, we did it on our personal computers because of time limitaions we could not set up the camera on google colab 
## Step 8.) real-time detections of our hand gestures
