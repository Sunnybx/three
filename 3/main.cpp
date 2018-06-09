#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
using namespace std;
using namespace cv;

float distance1(int x1,int y1,int x2,int y2)
{
	return sqrt(float((x1-x2)*(x1-x2)) + float((y1-y2)*(y1-y2)));
}

int main()
{
	Mat result = imread("binary009.bmp");
	imshow("binary", result);
	//Mat result;
	//blur(image, result, Size(3, 3));
	//Canny(image, result, 150, 100, 3);
	//imshow("edge.bmp", result);
	int rows = result.rows;
	int cols = result.cols;
	Mat zeros1 = Mat::zeros(rows,cols,CV_8UC1);
	float * a = new float[cols];
	float * b = new float[cols];
	for (int i = 0;i<cols;i++)
	{
		a[i] = 0;
		b[i] = 0;
	}
	int n = 0;	
	for (int i = 0;i < rows;i++)
	{
		uchar * data0 = result.ptr<uchar>(i);
		uchar * data1 = zeros1.ptr<uchar>(i);				
		int before = -1;
		int center;
		for (int j = 0;j < cols;j++)
		{					
			if (data0[j] == 255)
			{				
				if (before == -1)
				{
					before = j;
				}
				else
				{
					n = before;
					while(n <= j)
					{						
						a[n] = distance1(i,before,i,n)*40;
						b[n] = distance1(i,j,i,n)*40;
						if (a[n] < b[n])
							data1[n] = a[n];
						else
							data1[n] = b[n];
						n++;
					}
					before = j;
				}
			}			
		}
	}

	double *data = new double[rows*cols];
	for (int i = 0;i< rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			data[i*cols+j] = zeros1.at<uchar>(i,j);
		}
	}
	
	imshow("picture",zeros1);
	//imwrite("picture.bmp",zeros1);
	
	Mat * matrix = new Mat[rows * cols];
	for (int i = 0;i < rows * cols;i++)
	{
		matrix[i] = Mat::zeros(2,2,CV_64FC1);
	}
	
	for (int i = 1;i < rows-1;i++)
	{
		for (int j = 1;j < cols-1;j++)
		{
			matrix[i*cols+j].at<double>(0,0) = zeros1.at<uchar>(i+1,j) - 2 * zeros1.at<uchar>(i,j) + zeros1.at<uchar>(i-1,j);
			matrix[i*cols+j].at<double>(0,1) = zeros1.at<uchar>(i+1,j+1)+zeros1.at<uchar>(i,j)-zeros1.at<uchar>(i,j+1)-zeros1.at<uchar>(i+1,j);
			matrix[i*cols+j].at<double>(1,0) = zeros1.at<uchar>(i+1,j+1)+zeros1.at<uchar>(i,j)-zeros1.at<uchar>(i,j+1)-zeros1.at<uchar>(i+1,j);
			matrix[i*cols+j].at<double>(1,1) = zeros1.at<uchar>(i,j+1) - 2*zeros1.at<uchar>(i,j) + zeros1.at<uchar>(i,j-1);
		}
	}

	Mat * Value = new Mat[rows * cols];
	Mat * Vector = new Mat[rows * cols];
	Mat * gradient = new Mat[rows * cols];
	for (int i = 0;i < rows*cols;i++)
	{
		Value[i] = Mat::zeros(2,1,CV_64FC1);
		Vector[i] = Mat::zeros(2,2,CV_64FC1);
		gradient[i] = Mat::zeros(2,1,CV_64FC1);
	}
	for (int i = 0;i < rows*cols;i++)
	{
		eigen(matrix[i],Value[i],Vector[i]);
	}

	double temp0;
	Mat temp1 = Mat::zeros(2,1,CV_64FC1);
	for (int i = 0;i < rows * cols;i++)
	{
		if (Value[i].at<double>(0,0) > Value[i].at<double>(1,0))
		{
			temp0 = Value[i].at<double>(0,0);
			Value[i].at<double>(0,0) = Value[i].at<double>(1,0);
			Value[i].at<double>(1,0) = temp0;
			temp1.at<double>(0,0) = Vector[i].at<double>(0,0);
			temp1.at<double>(1,0) = Vector[i].at<double>(1,0);
			Vector[i].at<double>(0,0) = Vector[i].at<double>(0,1);
			Vector[i].at<double>(1,0) = Vector[i].at<double>(1,1);
			Vector[i].at<double>(0,1) = temp1.at<double>(0,0);
			Vector[i].at<double>(1,1) = temp1.at<double>(1,0);
		}
	}
	double * value = new double[rows*cols];
	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			value[i*cols+j] = Value[i*cols+j].at<double>(0,0);
		}
	}
	for (int i = 0;i < rows*cols;i++)
	{
		Vector[i] = Vector[i].t();
	}
	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			Vector[i*cols+j].at<double>(0,0) = (1.0/sqrt(Vector[i*cols+j].at<double>(0,0)*Vector[i*cols+j].at<double>(0,0)+Vector[i*cols+j].at<double>(0,1)*Vector[i*cols+j].at<double>(0,1)))*Vector[i*cols+j].at<double>(0,0);
			Vector[i*cols+j].at<double>(0,1) = (1.0/sqrt(Vector[i*cols+j].at<double>(0,0)*Vector[i*cols+j].at<double>(0,0)+Vector[i*cols+j].at<double>(0,1)*Vector[i*cols+j].at<double>(0,1)))*Vector[i*cols+j].at<double>(0,1);
			Vector[i*cols+j].at<double>(1,0) = (1.0/sqrt(Vector[i*cols+j].at<double>(1,0)*Vector[i*cols+j].at<double>(1,0)+Vector[i*cols+j].at<double>(1,1)*Vector[i*cols+j].at<double>(1,1)))*Vector[i*cols+j].at<double>(1,0);
			Vector[i*cols+j].at<double>(1,1) = (1.0/sqrt(Vector[i*cols+j].at<double>(1,0)*Vector[i*cols+j].at<double>(1,0)+Vector[i*cols+j].at<double>(1,1)*Vector[i*cols+j].at<double>(1,1)))*Vector[i*cols+j].at<double>(1,1);
		}
	}
	for (int i = 0;i<rows-1;i++)
	{
		for (int j = 0;j < cols-1;j++)
		{
			gradient[i*cols+j].at<double>(0,0) = zeros1.at<uchar>(i+1,j)-zeros1.at<uchar>(i,j);
			gradient[i*cols+j].at<double>(1,0) = zeros1.at<uchar>(i,j+1)-zeros1.at<uchar>(i,j);
		}
	}
	
	//cvtColor(zeros1, zeros1, CV_GRAY2RGB);
	//imshow("1", zeros1);
	Mat abc = Mat::zeros(rows, cols, CV_8UC1);
	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{

			if (Value[i*cols+j].at<double>(0,0)<0 
				&&
				(Vector[i*cols+j].at<double>(0,0)*gradient[i*cols+j].at<double>(0,0)+
				Vector[i*cols+j].at<double>(0,1)*gradient[i*cols+j].at<double>(1,0)) < 0
				&&zeros1.at<uchar>(i,j)>=80)
			{
				abc.at<uchar>(i,j) = zeros1.at<uchar>(i,j);
					//zeros2.at<Vec3b>(i,j)[0] = image0.at<Vec3b>(i,j)[0];
					//zeros2.at<Vec3b>(i,j)[1] = image0.at<Vec3b>(i,j)[1];
					//zeros2.at<Vec3b>(i,j)[2] = image0.at<Vec3b>(i,j)[2];						
			}			
		}
	}

	
	//imwrite("picture6.bmp",zeros1);
	cvtColor(abc, abc, CV_GRAY2RGB);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (abc.at<Vec3b>(i, j)[0] != 0 && abc.at<Vec3b>(i, j)[1] != 0 && abc.at<Vec3b>(i, j)[2] != 0)
			{
				abc.at<Vec3b>(i, j)[0] = 255;
				abc.at<Vec3b>(i, j)[1] = 0;
				abc.at<Vec3b>(i, j)[2] = 0;
			}				
		}
	}
	imshow("picture6", abc);
	delete[]Value;
	delete[]Vector;
	delete[]gradient;
	delete[]matrix;
	waitKey(0);
	return 0;
}