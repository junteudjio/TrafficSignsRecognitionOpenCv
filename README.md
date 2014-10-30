TrafficSignsRecognitionOpenCv
=============================




_____________________________________________________________INTELLIGENT  CAR  PROJECT______________________________________________________________

/* "Intelligent Car" is a project conducted in national school of applied sciences Tanger-Morocco during summer 2014 and more precisely within the
    Image Processing Laboratry.

   It's Goal is to aspire to an intelligent car able to detect and  recognize traffic  signs while moving.
	
   The detection Phase is done using The houghCircle function provided in OpenCv
   For the recognizing phase we have used a neural network classifier after having conducted a dimensionality reduction on the data Set using PCA.
*/


/*              _______author_______

   @author   :  TEUDJIO MBATIVOU Junior (Aspiring Data Scientist)
   @mail     :  teudjiombativou@gmail.com 
   @linkedin :  ma.linkedin.com/pub/junior-teudjio/8a/25b/3a1
*/


/* 		_______project tutor______
   
   @tutor   :  ABDELHAK Ezzine ( Professor at ENSA Tanger)
   @mail    :  ezzine.abdelhak@gmail.com
*/

 

/* 		_______DataSet Citation_______


   @Ref to the dataSet : http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
   
   J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. 
   In Proceedings of    the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.

   @inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    year = {2011},
    pages = {1453--1460}
    }   
*/


/* 		________READ ME FILE_______
  
   This file describe the "INTELLIGENT CAR PROJECT":

	1-  PROJECT FILES:

		a) 1_createDataSet.cpp : this code is used to construct the dataSet  in a .txt format from raw .ppm image files
		b) 2_shuffleDataSet.cpp : this code is used to shuffle the dataSet 
		c) 3_create_PCA_CovMatrix.cpp : this code is used to generate the PCA covariance matrix used for dimensionnality reduction
		d) 4_trainNeuralNet.cpp : this code is used to train a neural network for classifying road sign images and then save the generated model
		e) 5_recognition.cpp : this code is used to compute  traffic sign images recognition
		f) 6_detect_AND_recognize.cpp :  this code is used to compute both trafic sign image Detection & recognition
		g) CMakeLists.txt : CMake file use to execute each of these files.



	2-  REQUIREMENTS:
		
		a) OpenCV installed :  http://askubuntu.com/questions/334158/installing-opencv
		b) CMake installed :   http://www.cmake.org/install/ 

	

	3-  HOW TO RUN THE CODE:
		
		a) execute each piece of code of the project in the order they are listed in  Section (1):(PROJECT FILES)
		b) for that you will have to use the CMakeList.txt provided 
		

*/

