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
  
    this code is used to train a neural network for classifying road sign images and then save the generated model
*/



// global variable to store de probabilities for each class on each sample
cv::Mat classificationResult(1, CLASSES, CV_64F);


// global variable for the raw dataset of  size 4800=40*40*3
cv::Mat dataset4800(DATA_SET_SIZE,ATTRIBUTES,CV_64F);


// auxiliary function to convert an image vector to an image 3D matrix
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


// auxiliary function to shuffle the data set again if needed
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


//auxiliary function to read the dataset 
void read_dataset(char *filename, cv::Mat &data, cv::Mat &labels, cv::Mat &labs)
{
 
    Mat dataSet, classes;
    

    FileStorage fs(filename,FileStorage::READ);
    fs[SHUFFLE_DATA_SET_4800] >> dataSet ; fs.release();
	
    // randomly shuffle the matrix rows
    //MatShuffle(dataSet);
	


    	data = dataSet( Range::all(),Range(0,RESIZED_IMG_DIM) ).clone();
	labs = dataSet.col(RESIZED_IMG_DIM).clone();
	classes= labs;
	
	// construct the a label vector of size 13 for each image 
	// each of this vector has a unique cell sets to 1 (others=0) with it index corresponding to the image ClassId
	
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


// auxiliary function to compute the precision of the neural network
double precision(Mat & samples, Mat & samples_classes,  CvANN_MLP& nnetwork, bool show_sample_img = false)
{
	Mat sample_i;
	int correct_class=0, wrong_class=0;
	        // for each sample in the test set.
        for (int tsample = 0; tsample < samples.rows; tsample++) {
 
            // extract the sample
 
            sample_i = samples.row(tsample);
 
            //try to predict its class
 
            nnetwork.predict(sample_i, classificationResult);
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


		// if set to TRUE , show_sample_img indicates to show the image by the way while doing predictions
	    if( show_sample_img)
	    {	
		Mat img(IMG_NEW_DIM ,IMG_NEW_DIM ,CV_8UC3) ;

		double tab[RESIZED_IMG_DIM]; int m=0;
		for(int i=0; i< dataset4800.row(tsample+TRAINING_SAMPLES).rows ; i++)
		{
			for( int j=0; j< dataset4800.row(tsample+TRAINING_SAMPLES).cols; j++ )
			{	
				tab[m] = dataset4800.row(tsample+TRAINING_SAMPLES).at<double>(i,j); 
				
				m++;
			}	

		}
			
		convert(img, tab );
		imshow("image", img);		
		
		cout <<  " prediction :  " <<  maxIndex+1 << endl;
		waitKey(0);
		//cin.get();
	    }
	  
 
            //Now compare the predicted class to the actual class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            if ((int)samples_classes.at<double>(tsample, maxIndex)!=1)
            {
                // if they differ more than floating point error => wrong class
 
                wrong_class++;
 
                
 
            } else {
 
                // otherwise correct
 
                correct_class++;
            }
        }
 
return (double)correct_class*100/samples.rows ;

} 



int loadPCA(const string &file_name,cv::PCA& pca_)
{
    FileStorage fs(file_name,FileStorage::READ);
    fs["mean"] >> pca_.mean ;
    fs["e_vectors"] >> pca_.eigenvectors ;
    fs["e_values"] >> pca_.eigenvalues ;
    fs.release();

	return 1;//NYadd
}


/******************************************************************************/
 
int main( int argc, char** argv )
{
    

    cv::Mat labels = Mat::zeros(DATA_SET_SIZE, CLASSES, CV_64F);
    cv::Mat labs = Mat::zeros(DATA_SET_SIZE, 1, CV_64F);
 int n = 0; 


   //matrix to hold the training samples
    cv::Mat training_set(TRAINING_SAMPLES,ATTRIBUTES,CV_64F);
    //matrix to hold the training labels.
    cv::Mat training_set_classifications(TRAINING_SAMPLES,CLASSES,CV_64F);
 

    //matrix to hold the test samples
    cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_64F);
    //matrix to hold the test labels.
    cv::Mat test_set_classifications(TEST_SAMPLES,CLASSES,CV_64F);
 
    

    //load the training and test data sets.
    read_dataset(SHUFFLE_DATA_SET_FILE_4800, dataset4800, labels,labs);


    // load the pca cov_matrix	
    PCA pca;
    loadPCA( PCA_FILE, pca);


   // reduce the dimension of the data set to 388 (99% of the variance retained)
    Mat dataset388 = pca.project(dataset4800);
    //dataset4800.release();


   // split the dataSet into training and test sets
   training_set = dataset388(Range(0,TRAINING_SAMPLES) , Range::all() ).clone();
   test_set = dataset388(Range(TRAINING_SAMPLES , DATA_SET_SIZE ) ,  Range::all() ).clone();
	dataset388.release();

   // do the same with the assiocated labels
training_set_classifications = labels(Range(0,TRAINING_SAMPLES) , Range::all() ).clone();
   test_set_classifications = labels(Range(TRAINING_SAMPLES , DATA_SET_SIZE ) ,  Range::all() ).clone();
	labels.release();



 
        // define the structure for the neural network 
        // The neural network has 3 layers.
        // - one input node per attribute in a sample ---> 388 input nodes
        // - 500 hidden nodes
        // - 13 output node, one for each class.
 
       cv::Mat layers(3,1,CV_32S);
        layers.at<int>(0,0) = ATTRIBUTES;//input layer
        layers.at<int>(1,0)= HIDDEN;//hidden layer
        layers.at<int>(2,0) =CLASSES;//output layer
 
        //create the neural network.
        //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
        CvANN_MLP nnetwork(layers, CvANN_MLP::GAUSSIAN,0.6,1);

        CvANN_MLP_TrainParams params(                                   
 
                                        // terminate the training after either 1000 
                                        // iterations or a very small change in the
                                        // network wieghts below the specified value
                                        cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 60, 0.00001),
                                        // use backpropogation for training
                                        CvANN_MLP_TrainParams::BACKPROP, 
                                        // co-efficents for backpropogation training
                                        // recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
                                        0.0001, 
                                        0.1);
 
        // train the neural network (using training data)
 
        printf( "\nUsing training dataset\n");
        
	int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
        printf( "Training iterations: %i\n\n", iterations);
 
        // Save the model generated into an xml file.
        CvFileStorage* storage = cvOpenFileStorage( NEURAL_NET_FILE, 0, CV_STORAGE_WRITE );
        nnetwork.write(storage,NEURAL_NET);
        cvReleaseFileStorage(&storage); 
 
        
        // traing set precision
	double precision_train;

	// test set precision
	double precision_test;

	//precision_train = precision(training_set, training_set_classifications, nnetwork);
 	//cout << "precision training set "   <<  precision_train << endl;

	// compute test set error , and  display images  by the way.
	
	precision_test = precision(test_set, test_set_classifications, nnetwork);
	
 
       
  
   cout <<  "precision  test set  " <<   precision_test << endl;

   //cout <<  "diffirence  " <<   precision_train - precision_test << endl;

       
 
        return 0;
 
}
