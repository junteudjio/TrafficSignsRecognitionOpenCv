// TrainNetwork.cpp : Defines the entry point for the console application.
 
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv; 


#define IMG_NEW_DIM     40
#define RESIZED_IMG_DIM     4800
#define NUM_OF_CLASSES     13
#define HIDDEN    300
/******************************************************************************/
 
#define DATA_SET_SIZE 8729       //Number of samples in  dataset
#define ATTRIBUTES 388  //Number of pixels per sample.16X16

#define TRAINING_SAMPLES 6000       //Number of samples in test dataset
#define TEST_SAMPLES 2729       //Number of samples in test dataset
#define CLASSES 13                  //Number of distinct labels.
 

#define RAW_DATA_SET_4800  "dataset4800.txt"
#define SHUFFLE_DATA_SET_FILE_4800 "dataSetShuffle4800.yml"
#define SHUFFLE_DATA_SET_4800 "dataSetShuffle4800"
#define PCA_FILE "pca480.yml"
#define NEURAL_NET_FILE "neuralNet.xml"
#define NEURAL_NET "neuralNet" 


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


/* 		________code utility_______
  
    this code is used to compute  traffic sign images recognition
*/




// global variable to store de probabilities for each class on each sample
cv::Mat classificationResult(1, CLASSES, CV_64F);


// global variable for the raw dataset of  size 4800=40*40*3
    cv::Mat dataset4800(DATA_SET_SIZE,ATTRIBUTES,CV_64F);


void convert(cv::Mat &img,double pixelArray[])
{
 

int k=0;



for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        
	
	img.at<Vec3b>(i, j)[0] = (int)pixelArray[k]; 

	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img.at<Vec3b>(i, j)[1] = (int)pixelArray[k]; 
	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img.at<Vec3b>(i, j)[2] = (int)pixelArray[k]; 

	k++;

	
    }
	
}


}



void convertToVect(cv::Mat &img, Mat &  img_vect)
{
 

int k=0;



for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        
	
	img_vect.at<double>(0,k) = img.at<Vec3b>(i, j)[0] ; 

	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img_vect.at<double>(0,k) = img.at<Vec3b>(i, j)[1] ;
	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img_vect.at<double>(0,k) = img.at<Vec3b>(i, j)[2] ; 

	k++;

	
    }
	
}


}






void MatShuffle(Mat & dataset)
{
	vector< Mat > vv;
	Mat res;

	for( int i=0; i<dataset.rows; i++)
	{
		vv.push_back( dataset.row(i).clone());

	}
	
	random_shuffle(vv.begin(), vv.end());
	
	for(int i=0; i<vv.size();  i++)
	{
		res.push_back( vv[i] ) ;
	}
	dataset = res.clone();
	
	cout << "  shuffle over  " << endl;
	
}

void read_dataset(char *filename, cv::Mat &data, cv::Mat &labels, cv::Mat &labs)
{
 
    Mat dataSet, classes;
    

    FileStorage fs(filename,FileStorage::READ);
    fs[SHUFFLE_DATA_SET_4800] >> dataSet ; fs.release();
	
   // randomly shuffle the matrix rows
    MatShuffle(dataSet);
	


    	data = dataSet( Range::all(),Range(0,RESIZED_IMG_DIM) ).clone();
	labs = dataSet.col(RESIZED_IMG_DIM).clone();
	classes= labs;
	
	
	
	for(int i=0 ; i <  classes.rows ; i++)	
	{
		for( int j = 0; j < classes.cols ; j++) 
		{
			switch( (int)classes.at<double>(i,j) )
			{
			case 13 : labels.at<double>(i,0) = 1; break;
			case 14 : labels.at<double>(i,1) = 1; break;
			case 15 : labels.at<double>(i,2) = 1; break;
			case 17 : labels.at<double>(i,3) = 1; break;
			case 19 : labels.at<double>(i,4) = 1; break;
			case 20 : labels.at<double>(i,5) = 1; break;
			case 21 : labels.at<double>(i,6) = 1; break;
			case 27 : labels.at<double>(i,7) = 1; break;
			case 33 : labels.at<double>(i,8) = 1; break;
			case 34 : labels.at<double>(i,9) = 1; break;
			case 35 : labels.at<double>(i,10) = 1; break;
			case 36 : labels.at<double>(i,11) = 1; break;
			case 37 : labels.at<double>(i,12) = 1; break;
			default: break;
			}
		}
	}


    
}


double predict(Mat & sample,CvANN_MLP& nnetwork)
{
	
 
            nnetwork.predict(sample, classificationResult);
            /*The classification result matrix holds weightage  of each class. 
            we take the class with the highest weightage as the resultant class */
 
            // find the class with maximum weightage.
            int maxIndex = 0;
            double value=0.0;
            double maxValue=classificationResult.at<double>(0,0);
            for(int index=1;index<CLASSES;index++)
            {   value = classificationResult.at<double>(0,index);
                if(value>maxValue)
                {   maxValue = value;
                    maxIndex=index;
 
                }
            }

       
return maxIndex + 1;

} 



int loadPCA(const string &file_name,cv::PCA& pca_)
{
    FileStorage fs(file_name,FileStorage::READ);
    fs["mean"] >> pca_.mean ;
    fs["e_vectors"] >> pca_.eigenvectors ;
    fs["e_values"] >> pca_.eigenvalues ;
    fs.release();

	return 1;//NY
}


/******************************************************************************/
 
int main( int argc, char** argv )
{
PCA pca;

loadPCA(PCA_FILE, pca);

string img_path;

cout <<  "path to road sign image to recognize : ";
cin >> img_path;


while(img_path.c_str())
{

// get the image to recognize
Mat test_img =  imread(img_path);
imshow("img",test_img); 
waitKey(0);


Mat test_img_vect(1,RESIZED_IMG_DIM,CV_64FC1) ;

// resizing img to the standard size of 40*40*3
Mat resizedImg(IMG_NEW_DIM,IMG_NEW_DIM,CV_8UC3) ;
resize(test_img , resizedImg ,  resizedImg.size() );

// convert the image 3D matrix to a vector
convertToVect(  resizedImg, test_img_vect);

// apply PCA to that image to reduce it dimension ( 4800 ---> 388 )
    Mat test_img_388 = pca.project(test_img_vect);


 
        // define the structure for the neural network (MLP)
        // The neural network has 3 layers.
        // - one input node per attribute in a sample so 388 input nodes
        // - 500 hidden nodes
        // - 13 output node, one for each class.
 
       cv::Mat layers(3,1,CV_32S);
        layers.at<int>(0,0) = ATTRIBUTES;//input layer
        layers.at<int>(1,0)= HIDDEN;//hidden layer
        layers.at<int>(2,0) =CLASSES;//output layer
 
        //create the neural network.
        //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
        CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,1,1);
	

	// retrieve the neural network modeled previously
        // nnetwork.load("param.xml", "DigitOCR");
	nnetwork.load(NEURAL_NET_FILE, NEURAL_NET);
	int pred;

	
 
       pred = predict(test_img_388, nnetwork);
	
	cout << " predicted value :   " << pred<< endl << endl << endl;
	
	
	cout <<  " path to a new image: " ;
	cin >>  img_path ;
 
  }     
 
        return 0;
 
}
