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
[image6]: ./examples/webImages.png
[image7]: ./examples/first-conv-layer.png

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

### Training

To train the model, I used an Adam Optimizer. The training was done with 20 Epochs  and a batch size of 128. In order for the model to
generalize better, I used dropouts in the two fully connected layers before the output layer. The drop out probability was set to 0.5 during
the training.

My final model results were:
* training set accuracy of   :99.4%
* validation set accuracy of :97.9%
* test set accuracy of       :96.6%


The initial architecture I started with was the LeNet architecture. That gave around 94% validation accuracy without any data augnmentation. 
With the data augmnetaiton, the validation accuracy improved by ~2%. Fur further improvements, I added more complexity to the model by
increasing the number of features in the first and second convolutional layers. This resulted in increasing the validation set accuracy to ~98%

The model seems to generalize reasonably well giving ~97% accuracy on the test set.

### Testing  Model on New Images

Here are five German traffic signs that I found on the web that seem reasonably similar to images in the traning set.
	
![alt text][image6] 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)							| 
| Speed limit (70km/h)  | Speed limit (70km/h)							| 
| Speed limit (80km/h)  | Speed limit (80km/h)							| 
| Go straight or right  | Go straight or right							|
| Slippery Road			| Wild animals Crossing  						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
This seems a little less than the accuracy achieved on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very  sure that this is a speed limit 30 km/h sign (probability of 1.0)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)	 						| 
| ~0    				| Speed limit (50km/h)							|
| ~0					| Speed limit (70km/h)							|
| ~0	      			| Speed limit (20km/h)							|
| ~0				    | Yield   										|

The next three images also the model is very sure about the prediction with the most probable class having probability of ~1.0
For the last image, where the prediction was wrong, the probabilities are as below:

		  
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Bicycles crossing	 			    			| 
| 0.007    				| Wild animals crossing							|
| ~0					| Slippery Road							        |
| ~0	      			| Children crossing							    |
| ~0				    | Road narrows on the right   					|


### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The output of the first convolutional layer was visualized with the first traffic sign image from the web as the input. As can be seen from the
figure below, the layer seems to be activating on the edges of the speed limit letters and the circular outline.

![alt text][image7]