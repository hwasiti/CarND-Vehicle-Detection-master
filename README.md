# Udacity Self-Driving Car Engineer Nanodegree Program




### Vehicle Detection Project







The goals / steps of this project are the following:




* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.




## [Rubric](https://review.udacity.com/%23!/rubrics/513/view)  Points




### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.




#### [1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md)  is a template writeup for this project you can use as a guide and a starting point.




Here it is.




All of the code for the project is contained in the Jupyter notebook P4_submission.ipynb

## Histogram of Oriented Gradients (HOG)



1. **Explain how (and identify where in your code) you extracted HOG features from the training images**

The program loaded all the vehicle and non-vehicle images. Here are random samples of vehicle and non-vehicle images from the dataset used in this project: 

![Random samples of vehicle and non-vehicle images](https://github.com/hwasiti/CarND-Vehicle-Detection-master/raw/master/output_images/1_dataset_samples.jpg)




Extracting HOG features code is defined by the method get_hog_features and is contained in the cell titled "Lesson Functions", which contains all the helper functions that has been used in the lessons of project 5. The figure below shows a vehicle and a non-vehicle images and their corresponding Histogram of Oriented Gradients (HOG). 

![Vehicle and a non-vehicle images and their corresponding Histogram of Oriented Gradients](https://github.com/hwasiti/CarND-Vehicle-Detection-master/raw/master/output_images/2_HOG_example.jpg)

The method extract_features in the section titled "Extracting Features" will extract the HOG features from the images. The function accepts list of image paths and HOG parameters. 

2. **Explain how you settled on your final choice of HOG parameters.**

First, I fixed all the tuning parameters and tried to change only one parameter at time, in order to isolate the effect of this particular parameter on the Classifier's accuracy. I began to fiddle with the color space choices. I found that the YCrCb color space gave better results. Then tried to tweak the orientations number where I settled on 9, then explored different numbers for pix_per_cell and cell_per_block where I found that the best was 8 and 2 respectively. I used all HOG channels in the feature vectors. 

The total length of the feature vector was 6108. It took 14.59 seconds to train my SVC model with a test accuracy of 99%. 

I use the LinearSVC() from the sklearn.svm package. 

3. **Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**




I trained a linear Support Vector Machne (SVM) classifier with its default parameter using HOG features under the section titled "Train the Classifier'. I did not use the spatial intensity nor the channel intensity histogram and was able to achieve an accuracy around 99%.




## Sliding Window Search




1. **Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?**

Under the section titled "Implement a sliding-window technique and use your trained classifier to search for vehicles in images" I adapted the method find_cars from the lesson code example. This method extracts the HOG features for an input image with a sliding window search. Rather than extracting the HOG features for each sliding window, it performs more efficiently by extracting the HOG features of the image, and subsampling that for every sliding window. The method also performs the classifier prediction on the HOG features of each sliding window region and returns a list of rectangles, which correspond the windows that predicted to contain a car. 

I explored several configurations of sliding window sizes and positions. So I settled on this configurations which gave me the best results in the project video output: 

ystart = 400 ystop = 656 scale = 1.5 

Below figure showing an example image to predict the position of car in the image using the sliding window method. 

![predict the position of car in the image using the sliding window method](https://github.com/hwasiti/CarND-Vehicle-Detection-master/raw/master/output_images/3_sliding%20window%20search_example.jpg)

2. **Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?**




Here are some test images passed through the pipeline and resulted with the following predictions:

![test images passed through the pipeline](https://github.com/hwasiti/CarND-Vehicle-Detection-master/blob/master/output_images/7_%20extremities%20of%20each%20identified%20label.jpg)


![test images passed through the pipeline](https://github.com/hwasiti/CarND-Vehicle-Detection-master/raw/master/output_images/test3.png)


![test images passed through the pipeline](https://github.com/hwasiti/CarND-Vehicle-Detection-master/raw/master/output_images/test1.png)




## Video Implementation




1. **Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)**

Here's a link to my video result: 

[https://www.youtube.com/watch?v=_bLZBG72WrI&feature=youtu.be](https://www.youtube.com/watch?v=_bLZBG72WrI&feature=youtu.be) 

2. **Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**




Below figure showing an example image to predict the position of car in the image using the sliding window method.


![](https://github.com/hwasiti/CarND-Vehicle-Detection-master/blob/master/output_images/3_sliding%20window%20search_example.jpg)




Because there are several false positive results that happened occasionally in the video processed, it was better to apply a threshold on the detected cars' rectangles. The threshold will exclude any overlapping boxes that are less than in number from this threshold. This was performed by drawing a heatmap of the overlapping rectangles using the add_heat method, like the image below:

![](https://github.com/hwasiti/CarND-Vehicle-Detection-master/blob/master/output_images/4_Heatmap.jpg)





I have experimented with threshold values, and found that a threshold of five will give an acceptable low false positives throughout the video. 

Then the scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label: 

![](https://github.com/hwasiti/CarND-Vehicle-Detection-master/blob/master/output_images/6_assign_label_to_each_car.jpg)

And the final detected area is set to the extremities of the identified labels: 

![](https://github.com/hwasiti/CarND-Vehicle-Detection-master/blob/master/output_images/7_%20extremities%20of%20each%20identified%20label.jpg)


**Discussion** 

**1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?** 

The most difficult part of this project was tuning the parameters. A lot of try and error needed to be done to reach an acceptable level of false positives and accuracy. There are several methods that could be considered for future to enhance the accuracy. Integration detections from several previous frames could be able to decrease the outliers and false positives. 

The pipeline most likely will fail where vehicle differ largely from the samples of vehicles in the training set. Also extreme changes in lighting will affect the accuracy of this method, like in driving at night. 

2. believe that the best approach, given plenty of time to pursue it, would be to perform a smoothing method to the detection of the car within the last few frames. Outliers when happens usually they have a transient appearance in just one frame. Aggregating heat maps over a finite

amount of last few frames and applying higher threshold will give more robust approach to reject outliers.




Other thing to consider to improve the accuracy is the choice of the classifier. Using U-Net convolutional network has been reported to give better results. This is definitely an approach that I am eager to test soon.
