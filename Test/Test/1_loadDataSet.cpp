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
#define RAW_DATA_SET_4800  "dataset4800.txt"//加载数据
 
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
  
    this code is used to construct the dataSet  in a .txt format from raw .ppm image files

*/



//  Function to convert 3D image matrix into a vector
void convertToPixelValueArray(cv::Mat &img,int pixelArray[])
{
 

Mat_<cv::Vec3b>::iterator it = img.begin<cv::Vec3b>() ;
Mat_<Vec3b>::iterator itend = img.end<Vec3b>() ;

int k=0;



for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        Vec3b bgr = img.at<Vec3b>(i, j);

	pixelArray[k]= (int)bgr[0]; 

	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        Vec3b bgr = img.at<Vec3b>(i, j);

	pixelArray[k]= (int)bgr[1]; 

	k++;

	
    }
	
}

for(int i = 0; i < img.rows; i++)
{
    for(int j = 0; j < img.cols; j++)
    {
        Vec3b bgr = img.at<Vec3b>(i, j);

	pixelArray[k]= (int)bgr[2]; 

	k++;	
    }
	
}


}

// auxiliary function to build DataSet text file
string convertInt(int number, char * prefix = "" , char * suffix = "")
{
    stringstream ss;//create a stringstream
    ss << prefix << number << suffix;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}


int main()
{
	//each number below represents the name of folder containing a set of images from the same road sign
	int trainingSample[NUM_OF_CLASSES] = {13,14,15,17,19,20,21,27,33,34,35,36,37}; 

	string rowPath = "./Images";

	// the name of text file containing the final dataSet
	string outputfile = RAW_DATA_SET_4800;

	fstream dataSet(outputfile.c_str(),ios::out);
	int pixelVector[RESIZED_IMG_DIM];


	// iterate through the different folder to fetch its images 
	for (int i=0 ; i< NUM_OF_CLASSES ; i++)
	{
		/* the next 4 are used to construct the path to .cvs file 
		   where are registered paths to road sign's images and 
		   other metadata
		*/
		
		string  numFolder  = convertInt(trainingSample[i], "000");
		string folder = rowPath + "/" + numFolder;
		string csvFile = folder + "/" + "GT-" + numFolder + ".csv" ;
		std::ifstream file(csvFile.c_str());

		// string to fetch each line of the previous file  
		std::string line;
		int numeroLigne = 0;
		
		// iterate through the file and fetch each image's metadata
		while (std::getline(file, line))
		{
			numeroLigne ++;
			
			if(numeroLigne == 1) continue;
			std::replace(line.begin(), line.end(), ';', ' ');

  			std::istringstream iss(line);
  			string rawInfo[8];
			string cell;
			int k=0;
			
			// string to fetch the road sign path
			string imagePath;
			
			// integers to fetch other metadata ( exact location coordinates and image's classId )
			int RoiX1 ,  RoiY1, RoiX2, RoiY2, ClassId; 

  			while (iss >> cell)
  			{
				rawInfo[k] = cell;
				k++;				
 			}
			
			imagePath =  folder + "/"  +  rawInfo[0] ;
			RoiX1 = atoi(rawInfo[3].c_str());
			RoiY1 = atoi(rawInfo[4].c_str());
			RoiX2 = atoi(rawInfo[5].c_str());
			RoiY2 = atoi(rawInfo[6].c_str());
			ClassId = atoi(rawInfo[7].c_str());
		
			//cout <<  imagePath <<  "   " <<  RoiX1 << "  " << RoiY1<<  "   " <<  RoiX2 << "  " << RoiY2 << "  "  << ClassId <<  endl;
				
			// loading img
			Mat img = imread( imagePath.c_str() , CV_LOAD_IMAGE_COLOR );
			

			
			// cropping img to get the exact image
			Rect ROI(RoiX1, RoiY1, RoiX2 - RoiX1, RoiY2 - RoiY1);
						
			Mat croppedImg = img(ROI).clone();

			
			// resizing img to  standardize their sizes to 40*40*3
			Mat resizedImg(IMG_NEW_DIM,IMG_NEW_DIM,CV_8UC3) ;
			resize(croppedImg , resizedImg ,  resizedImg.size() );

			//  free memory 
			img.release();
			croppedImg.release();

			
			// matrix img  to vector img  ( 40*40*3  ----> 4800*1 )
			
			convertToPixelValueArray( resizedImg , pixelVector );

			for( int l=0 ; l < RESIZED_IMG_DIM ; l++)
			{	
				dataSet << pixelVector[l] << " ";
			}

			// save the dataSet in a file.
			dataSet << ClassId << "\n";
				
		}		
	}

	
	// close file pointer to free memory
	dataSet.close();
    
    return 0;
}
