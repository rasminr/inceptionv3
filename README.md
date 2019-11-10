# inceptionv3
Ready made codes to train inception network 

The folllowing codes make it easier to train and test the inception network on a ubuntu system.

The ouput graph file is tested on a raspberry pi with tensorflow version 1.13


Environment:
Ubuntu 16.04
Python3.5.2
Tensorflow version : 1.14.0

Create an empty directory called photocreate.

The photocreate folder should be inside the /home/inceptionv3 folder, it is where the images are stored, with folder names as class/type of image.

Run the below command to train the inception network with the images.

python3 retrain.py --image_dir ./photocreate --output_labels labels.txt --output_graph output_graph.pb --how_many_training_steps 1500

The above command uses the python3 appllication to open the file retrain.py, which is the file in which scripts for the training are written. --image_dir show the directory in which the folders labelled as the group categories are named.
