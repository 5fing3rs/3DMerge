#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <ctype.h>

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{

    Mat img1 = imread("Football1.jpg"); 
    Mat img2 = imread("Football2.jpg"); 
    namedWindow("I2", WINDOW_NORMAL); namedWindow("I1", WINDOW_NORMAL); 
    imshow("I1",img1);
    imshow("I2",img2);

    Ptr<ORB> o1 = ORB::create();
    Ptr<ORB> o2 = ORB::create();
    vector<KeyPoint> pts1,pts2;
    Mat desc1,desc2;
    vector<DMatch> matches;

    Rect r1(1720,40,200,1040);
    Rect r2(0,0,150,1080);
    Mat mask1 = Mat::zeros(img1.size(),CV_8UC1);
    Mat mask2 = Mat::zeros(img1.size(),CV_8UC1);
    mask1(r1)=1;
    mask2(r2)=1;
    o1->detectAndCompute(img1,mask1,pts1,desc1);
    o2->detectAndCompute(img2,mask2,pts2,desc2);
    BFMatcher descriptorMatcher(NORM_HAMMING,true);

    descriptorMatcher.match(desc1, desc2, matches, Mat());
    // Keep best matches only to have a nice drawing.
    // We sort distance between descriptor matches
    Mat index;
    int nbMatch=int(matches.size());
    Mat tab(nbMatch, 1, CV_32F);
    for (int i = 0; i<nbMatch/2; i++)
    {
        tab.at<float>(i, 0) = matches[i].distance;
    }
    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    vector<DMatch> bestMatches;
    vector<Point2f> src,dst;
    for (int i = 0; i < nbMatch/2; i++)
    {
        int j = index.at<int>(i,0);
        cout << pts1[matches[j].queryIdx].pt<<"\t"<<pts2[matches[j].trainIdx].pt<<"\n";
        src.push_back(pts1[matches[j].queryIdx].pt+Point2f(0,img1.rows)); // necessary offset 
        dst.push_back(pts2[matches[j].trainIdx].pt);
    }
    cout << "\n";
    Mat h=findHomography(src,dst,RANSAC);
    Mat result;
    cout<<h<<endl;

    warpPerspective(img2, result, h.inv(), Size(3*img2.cols +img1.cols , 2*img2.rows+img1.rows));

    Mat roi1(result, Rect(0, img1.rows, img1.cols, img1.rows));
    img1.copyTo(roi1);
    namedWindow("I3", WINDOW_NORMAL); 
    imshow("I3",result);
    imwrite("result.jpg",result);
    waitKey();
   return 0;
}