# Traffic Sign Recognition

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Distribution-Training-Set.png "Dist-Train"
[image2]: ./examples/Distribution-Validation-Set.png "Dist-Valid"
[image3]: ./examples/Distribution-Test-Set.png "Test-Train"
[image4]: ./examples/grayscale-conversion.png "GrayScaleConversion"
[image5]: ./examples/augmentation.png
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 
Here is a link to my [project code](https://github.com/iyerhari5/P2-TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb)

Data Set Summary & Exploration

Here are some summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32xx
* The number of unique classes/labels in the data set is 43

The images below show the distribution of the classes in the training, validation and 
test data sets. As we can see, the distribution is not very uniform and there are some classes
that are not very well represented. However the distribution seems to be similar across
the training,validation and test datasets

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Model Architecture 

The original images in the data set are color images of size 32x32. Based on results reported in the literature, I decided to
convert the images to grayscale as the first step. This helps to reduce the dimensionality of the input space. The images are then
normalized by a simple transformation to center the data.

image = (image-128.0)/128.0

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image4]

Data Augmentation

As can be noted, the training set contains only around 35K images. In order to make the traning more generalizable, I decided to 
augment the data with samples generated from the training set itself. For this I implemented functions to add translation, rotation, zooming
and perspective projection on the images.

Here is an example of an original image and 4 more images generated with the described transformations from the original image.

![alt text][image5]

The augmented dataset hence should be more robust to differences in the pose of the camera, centering and rotation in the images 
presented to the neural network.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   			    	| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36 					|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Fully connected		| outputs 43        							|
|:---------------------:|:---------------------------------------------:| 

### Training

To train the model, I used an Adam Optimized. The training was done with 20 Epochs  and a batch size of 128. In order for the model to
generalize better, I used dropouts in the two fully connected layers before the output layer. The drop out probability was set to 0.5 during
the training.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of  97.9%
* test set accuracy of 96.6%


The initial architecture I started with was the LeNet architecture. That gave around 92% test accuracy without any data augnmentation. 
With the data augmnetaiton, the test accuracy improved by ~2%. Fur further improvements, I added more complexity to the model by
increasing the number of features in the first and second convolutional layers. This resulted in increasing the test set accuracy to ~97%



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


