#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;
    Mat img = imread("./img/demo2.jpg");  
    namedWindow("原图",0);
    resizeWindow("原图", 640, 480); 
    imshow("原图", img);

    //灰度化
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); 
    cvtColor(gray, img, COLOR_GRAY2BGRA);
    namedWindow("灰度化", 0);
    resizeWindow("灰度化", 640, 480); 
    imshow("灰度化", gray);

    //二值化
    Mat binary;
    threshold(gray, binary, 100, 200, THRESH_BINARY);
    namedWindow("二值化", 0);
    resizeWindow("二值化", 640, 480); 
    imshow("二值化", binary);

    //腐蚀
    Mat ercode;
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(13, 13));
    erode(binary, ercode, erodeElement);
    namedWindow("腐蚀", 0);
    resizeWindow("腐蚀", 640, 480);
    imshow("腐蚀", ercode);

    //轮廓检测
    vector<vector<Point> > contours;  //定义一个容器来存储所有检测到的轮廊
    vector<Vec4i> hierarchy;
    //轮廓检测函数
    Mat conv(ercode.size(), CV_8UC1);
    ercode.convertTo(conv, CV_8UC1);
    findContours(conv, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
    int index = 0;
    long max_Index = static_cast<long>(contours.size());
    if (index < max_Index - 1) {
        ++index;
    }

    //取出身份证号码区域
    vector<Rect> rects;
    Rect numberRect = Rect(0, 0, 0, 0);
    vector<vector<Point> >::const_iterator itContours = contours.begin();

    for (int i = 0; i <max_Index; ++i) {
        Rect rect = boundingRect(itContours[i]);
        numberRect = rect;

        // 剔除尺寸过小的区域(64*64)
        if (numberRect.height < 64 || numberRect.width < 64 ) {
            continue;
        }

        Mat dst(numberRect.height, numberRect.width, CV_8UC4, numberRect.area());
        dst = img(numberRect);

        char filename[25];
        sprintf(filename, "./output/region-%d.jpg", i);
        //imshow(name, dst);
        imwrite(filename, dst);

    }


    waitKey(0);
    return 0;
}
