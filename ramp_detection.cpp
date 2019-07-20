#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <iostream>
#include <bits/stdc++.h>
//#define tolerance 2
//#define threshold 100
#define blocked 80
#define constantSubtracted -35
#define neighbourhoodSize 25
#define medianBlurkernel 7


using namespace cv;
using namespace std;

Mat blueChannelProcessing(Mat img)
{
    Mat channels[3];
    //Splitting the image into 3 Mats with 1 channel each
    split(img, channels);   
    Mat b = channels[0];

    GaussianBlur(b , b, Size( 9, 9), 0, 0);     //Based on observation
    adaptiveThreshold(b,b,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,neighbourhoodSize, constantSubtracted);
    medianBlur(b,b,medianBlurkernel);   //To remove salt & pepper noise

    return b;

}

//2B - G channel
Mat twob_gChannelProcessing(Mat img)
{    
    Mat channels[3];
    split(img, channels);
    Mat fin = 2*channels[0] - channels[1];

    adaptiveThreshold(fin, fin,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY, neighbourhoodSize, constantSubtracted);
    medianBlur(fin, fin,medianBlurkernel);

    return fin;
}

//2B - R channel
Mat twob_rChannelProcessing(Mat img)
{
    Mat channels[3];
    split(img, channels);
    Mat fin = 2*channels[0] - channels[2];

    //For storing square of channels
    Mat_<int> b2, g2, r2, mean2;

    multiply(channels[0], channels[0], b2); //multiplies matrices
    multiply(channels[1], channels[1], g2);
    multiply(channels[2], channels[2], r2);

    Mat_<int> mean = (Mat_<int>)((channels[0] + channels[1] + channels[2])/3);
    multiply(mean, mean, mean2);
    
    Mat_<int> zero_moment = (Mat_<int>)(b2 + g2 + r2)/3;
    
    //Variance is being used purely based on observation
    Mat_<float> variance = (Mat_<float>)((zero_moment - mean2)/5);
    //Vairance= (Mean of sq.s) - (sq. of mean)

    Mat mask, result1;
    /*for (int i=0;i<variance.rows;i++){
    	for(int j=0;j<variance.cols;j++){
    		cout << (int)variance.at<uchar>(i,j) << " " ;
    	}
    	cout <<endl;
    }
    */
    threshold(variance, mask, 254, 255, THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    imshow("mask",mask);

    //Taking intersection of original & thresholded variance intensities 
    bitwise_and(fin, mask, result1);

    adaptiveThreshold(result1,result1,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,neighbourhoodSize, constantSubtracted);

    medianBlur(result1, result1, medianBlurkernel);

    return result1;

}

/*int a[4] = {1,0,-1,0},b[4] = {0,1,0,-1};

bool isValid(int i,int j,Mat img){
	if(i<img.rows&&i>=0&&j<img.cols&&j>=0) return true;
	else return false;
}

int dfs(Mat img,int i,int j,Mat occupied,int fill)
{*/
	//cout << "1"<<endl;
	/*queue<Point> q;
	q.push(Point(i,j));
	int count =1;long long int avgi,avgj;
	while(!q.empty()){*/
		//cout << i << " " << j << endl;
		/*if(occupied.at<uchar>(i,j)) {
			q.pop();
			continue;
		}*/
		/*i=q.front().x;j=q.front().y;*/
		//cout <<i<<" "<<j<<endl;
		
		/*occupied.at<uchar>(i,j) = fill;
		
		q.pop();
		
		for(int k=0;k<4;k++){*/
			//cout << i+a[k] <<" " << j+b[k] <<endl;
			//cout << k <<endl;
			//cout << (isValid(i+a[k],j+b[k],img)) /*&& !occupied.at<uchar>(i+a[k],j+b[k]))*/ <<endl;
			/*if(isValid(i+a[k],j+b[k],img) && !occupied.at<uchar>(i+a[k],j+b[k]) && (img.at<Vec3b>(i+a[k],j+b[k])[0]-img.at<Vec3b>(i,j)[0])<tolerance)
			{
				count++;
				avgi+=i+a[k];
				avgj+=j+b[k];
				occupied.at<uchar>(i+a[k],j+b[k]) = fill;
				q.push(Point(i+a[k],j+b[k]));
			}*/
			//cout << "yes" <<endl;
			
	/*	}
	}*/
	//cout << i << " " << j << endl;
	/*cout<<count<<endl;
	return count;
	avgi/=count;avgj/=count;
	if(count > threshold)
	{
		if(avgj>img.cols*2/5&&avgj<img.cols*3/5)
		{
			cout << "Ramphole Detected" << endl;
		}
	}
}*/
int main(int argc,char** argv) 
{
	//VideoCapture vid (argv[1]);
	/*namedWindow( "win" ,0 );
	namedWindow( "win1" ,0 );
	while(1)
		{
			Mat imge = imread(argv[1]);
			//vid >> imge;
			Mat img = imge.clone();
			Mat hsv[3]; 
			cvtColor(imge, img, COLOR_BGR2HSV);
			split(img,hsv);
			imshow( "win1" ,hsv[0]);
			imshow( "win" ,img );
			Mat occupied(img.rows,img.cols,CV_8UC1,Scalar(0));
			int fill = 0;
			for(int i=0;i<img.rows;i++)
			{
				for(int j=0;j<img.cols;j++)
				{
					
					if(!occupied.at<uchar>(i,j))
					{
						fill+=1;
						//cout <<fill<<endl;
						fill%=255;
						if(dfs(img,i,j,occupied,fill)<50) fill--;;
						
					}

					//cout << "y"<<endl;
					//return 0;
					//waitKey(0);
					//cout <<"done"<<endl;
				}
			}
			cout <<"finished"<<endl;
			cout <<fill<<endl;
			imshow("final",occupied);
		
			waitKey(0);
		}*/
	Mat img = imread(argv[1]);
	//Mat imge;
	Mat hsv[3];
	Mat result;
	Mat temp,twobg,twobr,bl,intersection;
	
	result = Mat(img.rows,img.cols,CV_8UC1,Scalar(0));
	twobg = twob_gChannelProcessing(img); 
	cout <<1 <<endl;
	twobr = twob_rChannelProcessing(img);
	cout << 2 <<endl;
	bl = blueChannelProcessing(img);
	cout << 3 <<endl;
	intersection = Mat(img.rows,img.cols,CV_8UC1,Scalar(0));
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++)
		{
			//cout << i <<endl;
			intersection.at<uchar>(i,j) = (twobg.at<uchar>(i,j) && twobr.at<uchar>(i,j) && bl.at<uchar>(i,j))*255;
			
		}
	}
	cout << "done " <<endl;
	medianBlur ( intersection, result, 9);
	imshow("After_application_of_OTSU",result);
	imshow("Original",img);
	imshow("2B-G",twobg);
	imshow("2B-r",twobr);
	imshow("B",bl);
	imshow("intersection",intersection);
	waitKey(0);	
	//return 0;
}