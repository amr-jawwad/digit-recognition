Digit recognition service
=========================
Developed by: Amr Jawwad (July,2017)
=========================

Intro
-----
This program receives an image of a digit (0-9), recognizes it, and outputs its numeric value.
The solution implemented uses a machine learning model to address the task.
The model has a prediction accuracy of ~99.8% on the MNIST dataset(http://yann.lecun.com/exdb/mnist/)
An example file (Input.json)is given to show the program's capability of correctly-classified images, even though some of them are significantly different from the training data. (See Example.png)

How to use
----------
Using the main service:
	The program reads JSON input from a JSON file given to the function DigitRecogService in MainService.py. A sample file (Input.json) is shown for example.
	The program expects the JSON input to contain URL(s) to images of digits.
	For the program to run, you need to have:
		1- Input JSON file
		2- Connection to the internet (to download the input images)
		3*- A pre-trained model file (*.pkl)
	If you have these prerequisites, just give the JSON file name as input to the DigitRecogService function, and you'll find your output in another JSON file Output.json in the working directory of the program.

If you don't have the pre-trained model file, you should build the model first,
Building the model:
	To build just run the function BuildingModel.py, it doesn't have any inputs.
	If the function runs successfully, you should find the model file (SVM_Model.pkl) in the working directory of the program.
	Take care that the function could take some time to fit the model, depending on your computer's capabilities.

Take care:
	This program deals (reads/writes) with files, so please make sure you have the necessary permission to do so in the working directory of the program.
	

Technical description
---------------------
The program is divided into two files:
	1- The Model Builder: (BuildingModel.py)
		Builds the SVM Machine Learning Model.
	2- The Main Service: (MainService.py)
		JSON in/out service, that takes URL(s) of images of digits and outputs the recognized numerical values.

The SVM:
--------
	This program trains a SVM (Support Vector Machine) to recognize digits using MNIST training dataset.
	The dataset used consisted of 70,000 (28x28) images of digits (0-9), with the following frequencies:
	0: 6903		1: 7877		2: 6990		3: 7141		4: 6824
	5: 6313		6: 6876		7: 7293		8: 6825		9: 6958
	For training the model, the dataset was split between: training data (80% or 56,000 images), and testing data (20% or 14,000 images).
	The split was done on a stratified random basis to ensure balance between classes in both sections.
	The hyperparameters of the SVM were chosen after multiple trials (and similar applications review) to be:
	Kernel function:	RBF (Radial Basis Function)
	Gamma: 				0.05
	C:					5
	
	In BuildingModel.py, there is a deactivated code to run an exhaustive grid search in the hyperparameter space using cross validation, for the values of:
	Kernel functions:	RBF, Linear
	Gamma:				0.05, 1e-3, 1e-4
	C:					1, 5, 10, 100, 1000
	In this scenario, all the possible combinations are tried out, and the best hyperparameters are selected based on the precision score.
	To activate this part of the code:
		1- Open BuildingModel.py.
		2- Find this line: 	Get_Optimal_Model_Params = 0
		3- Change it to  :	Get_Optimal_Model_Params = 1
	Take care that running the exhaustive grid search could take very long time (in the order of days).
	
	Model Performance:
	------------------
	Classification Report:
					 precision    recall  f1-score   support

		Class   0.0       1.00      1.00      1.00      1381
				1.0       1.00      1.00      1.00      1575
				2.0       1.00      1.00      1.00      1398
				3.0       1.00      1.00      1.00      1428
				4.0       1.00      1.00      1.00      1365
				5.0       1.00      1.00      1.00      1263
				6.0       1.00      1.00      1.00      1375
				7.0       1.00      1.00      1.00      1458
				8.0       0.99      1.00      1.00      1365
				9.0       1.00      0.99      1.00      1392

		avg / total       1.00      1.00      1.00     14000
	
	Confusion Matrix:
		0    1  2  3  4  5   6  7  8  9
	0	1377, 0, 0, 0, 0, 0, 1, 0, 1, 2
	1	0, 1574, 0, 0, 0, 0, 0, 0, 1, 0
	2	0, 0, 1395, 0, 0, 0, 0, 1, 2, 0
	3	0, 0, 2, 1421, 0, 2, 0, 2, 1, 0
	4	0, 0, 0, 0, 1364, 0, 0, 0, 0, 1
	5	0, 0, 0, 1, 0, 1260, 2, 0, 0, 0
	6	0, 0, 0, 0, 0, 0, 1374, 0, 1, 0
	7	0, 0, 3, 1, 0, 0, 0, 1453, 1, 0
	8	0, 1, 0, 0, 0, 1, 0, 0, 1363, 0
	9	1, 0, 0, 1, 2, 1, 0, 0, 2, 1385
	
	Model Accuracy: 0.997571428571

Digit Recognition Service:
--------------------------
	The service reads the JSON input, and downloads the image(s) from the given URL(s).
	The service then loads the pre-trained model file SVM_Model.pkl which should in the working directory of the program.
	Image Pre-processing:
	--------------------
		Some basic image pre-processing is done on the input images before giving them to the model for prediction.
		This includes:
			1- Image resizing to 28x28 (the MNIST image size).
			WARNING: Resizing doesn't maintain aspect ratio, so images with aspect ratios very far from square will be severely deformed which might affect the model's accuracy.
			2- Converting to grayscale
			3- Inverting the image if needed:
				The model is trained on images which have white digits on black backgrounds.
				ASSUMING most of the image is background, using the histogram of the brightness values of the image, if it was found that the majority of the pixels are above 127 in brightness (0 is black and 255 is white), the image is inverted.
	After pre-processing the image is forwarded to the model for prediction, and the result added to the list which will be converted into the JSON output eventually.

Future Work:
------------
	1- Implementing a REST API, instead of dealing with JSON files.
	2- More advanced image pre-processing, such as:
		- Taking into consideration only the bounding box of the digit in the image
		- More advanced background detection and removal
	3- Since the program was designed to be modular, very little work could be done to MainService.py to use any pre-trained model (not necessarily SVM).
