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
#define PCA_FILE "pca480.yml"



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
  
    this code is used to generate the PCA covariance matrix used for dimensionnality reduction (99% of the variance is retained here)
*/




void savePCA(const string &file_name,cv::PCA& pca_)
{
    FileStorage fs(file_name,FileStorage::WRITE);
    fs << "mean" << pca_.mean;
    fs << "e_vectors" << pca_.eigenvectors;
    fs << "e_values" << pca_.eigenvalues;
    fs.release();
}


int main ()
{
    Mat dataset;

    // load the shuffled dataSet  ( 8729(rows)  *  48001(cols) )  the last column for the image ClassId	
    FileStorage fs(SHUFFLE_DATA_SET_FILE_4800,FileStorage::READ);
    fs[SHUFFLE_DATA_SET_4800] >> dataset ;

    // exclude the ClassId before performing PCA
    Mat data = dataset(Range::all(), Range(0,4800));


    //  perform to retain 99%  of the variance
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW , 0.99f);

   // save the model generated for  future uses.
    savePCA(PCA_FILE, pca); 

    return 0;

}
