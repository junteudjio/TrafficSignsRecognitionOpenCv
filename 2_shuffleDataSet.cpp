#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
#define RESIZED_IMG_DIM     4800
#define RAW_DATA_SET_4800  "dataset4800.txt"
#define SHUFFLE_DATA_SET_FILE_4800 "dataSetShuffle4800.yml"
#define SHUFFLE_DATA_SET_4800 "dataSetShuffle4800"


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
  
    this code is used to shuffle the dataSet 
*/


// convert image Vector to image 3D matrix
void convert(cv::Mat &img,int pixelArray[])
{
 

Mat_<cv::Vec3b>::iterator it = img.begin<cv::Vec3b>() ;
Mat_<Vec3b>::iterator itend = img.end<Vec3b>() ;

int k=0;



for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        

	img.at<Vec3b>(i, j)[0] = pixelArray[k]; 

	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img.at<Vec3b>(i, j)[1] = pixelArray[k]; 
	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        	img.at<Vec3b>(i, j)[2] = pixelArray[k]; 

	k++;

	
    }
	
}


}


int main ()
{


	// raw dataset file  8729(rows) * 4800(cols)  not yet shuffle  
	std::ifstream file(RAW_DATA_SET_4800);
	std::string line;
	
	Mat dataSet;
	int ligne =0;

	// vector of vector containing each line of the dataset file = each image pixels (1*4800)
	vector< vector<double> > vv;


	// iterates through the file to construct the vector vv
	while (std::getline(file, line))
{
  std::istringstream iss(line);
  double n;
	int k = 0;
	double tab[ RESIZED_IMG_DIM +1 ];
	vector<double> v;
 	


  while (iss >> n)
	{ 	
		if( k == RESIZED_IMG_DIM +1) break;
		//tab[k] = n;  
		v.push_back(n);
		k++;
	}
	

		//Mat img(1,RESIZED_IMG_DIM +1,CV_64F,tab);
		//dataSet.push_back(img);
		vv.push_back(v);
		ligne ++ ;
	
}


// finaly we can randomly shuffle the dataSet
random_shuffle(vv.begin(), vv.end());



// save the randomized dataSet back to a file
for( int i=0; i < vv.size(); i++)
{ 

	double* tab = &vv[i][0];
	Mat img(1,RESIZED_IMG_DIM +1,CV_64F,tab);
	dataSet.push_back(img);
}
FileStorage fs(SHUFFLE_DATA_SET_FILE_4800,FileStorage::WRITE);   
fs<< SHUFFLE_DATA_SET_4800 << dataSet;
fs.release(); 

}
