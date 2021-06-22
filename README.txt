DL-Model-Flask-Deployment

Prerequisites You must have Scikit Learn, Keras , Tensorflow , Numpy, Pandas (for Machine Leraning Model/Deeplearning) and Flask (for API) installed.

Project Structure This project has four major parts :

model.h5 - This contains code for or deep learning model to predict the name(character)/emotion of a given image  based on training data in which consist of images of different characters of the seriel given 

app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value. 

Run app.py using below command to start Flask API python app.py By default, flask will run on port 5000.

Navigate to URL http://localhost:5000 You should be able to view the homepage as below : alt text

Enter an image from the loacal system and hit Predict.

NOTE: RUN ALL THE LINES FROM THE IPYNB FILE 

you should be able to see the predcited vaule on the HTML page.
-------------------------------------------------------------------
Steps to follow on colab:
1)download the given folder and also download the video in which we want to implement the image and emotion detection on
2)open the colab file and run the Mspa_project_phase1file line by line make sure you give your drive/local path for the path of the video file 
3)once the steps are done you will be able to downlaod a file which will have images clustered into different files 
4)with some manual effort rename the files and remove unwanted file/images which are clustered into differnet files based on the character names
5)once done with this processing upload the given file back to your working directory 
6)now we will use svm and Deep neural network model for prediction 
7)once we fit the model for the built neural network downlaod the h5.file for further process
**NOTE : i have named face_extracted in training and test set folder which is as same as unknowns in the same folder
can change the name and fit the model once again and do the predictions.
()()()()()()()()()---------->
steps to follow on spyder:
1) Set the given detection file as working directory
2)open the app.py file and change the model.h5 file path to the local path of your system and run the file
3)**open anaconda prompt and enter into the path where the app.py file is running using (cd) and run the file as (python app.py)***
4)once the execution is done we will get a ip address copy the ip address and paste on the web browser 
5)Randomly choose an img file from local dir or use the images from face_extracted folder from the Detections folder 
5)now the prediction model will be able to predict the images / emotion

****Note: if the page keeps reloading please check the anaconda console and refresh the page and run again***


Folder Contents:
---->
Mspa_project.ipynb file

In the above file we have implemented Image detection and Classification along with emotion Detection for the given 
images and also have tried few image segmentation techniques
The pretrained models used in the code are Mtcnn(face detection) and Rmn(emotion detection)

Platform used: Google colab
programming language:python 
---->
app.py
index.html
base.html
.CSS and JS file 
uplaods folder

Platform: spyder
programminng Lang: Python,Html,Js and CSS


_____________________#%%$#%$^%$%_______________________THANK____YOU_______________________%#^&@#^$#^@#%___________________