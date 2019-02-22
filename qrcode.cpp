#include <zxing/common/Counted.h>
#include <zxing/Binarizer.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/ReaderException.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/Exception.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/MatSource.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>

using namespace cv;
using namespace std;
using namespace zxing;
using namespace zxing::qrcode;

static const char *short_options = "f:o:h";
static const struct option long_options[] = {
    {"file", required_argument, NULL, 'p'},
    {"out", required_argument, NULL, 'o'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

bool createDir( const char *szDirectoryPath , int iDirPermission = 0744 ){

    if ( NULL == szDirectoryPath ){
        return false;
    }

    const int iPathLength = static_cast< int >( strlen( szDirectoryPath ) );

    if (iPathLength > PATH_MAX ){
        return false;
    }

    char szPathBuffer[ PATH_MAX ] = { 0 };
    memcpy( szPathBuffer , szDirectoryPath , iPathLength );

    for ( int i = 0 ; i < iPathLength ; ++i ){
        char &refChar = szPathBuffer[ i ];
        //目录分隔符
        if ( ( '/' == refChar ) && ( 0 != i ) ){
            refChar = '\0';
            //判断当前目录是否存在
            int iStatus = access( szPathBuffer , F_OK );
            if ( 0 != iStatus ){
                if ( ( ENOTDIR == errno ) || ( ENOENT == errno ) ){
                    //以指定权限创建目录
                    iStatus = mkdir( szPathBuffer , iDirPermission );
                    if ( 0 != iStatus ){
                        return false;
                    }
                } else {
                    return false;
                }
            }
            refChar = '/';
        }
    }
    return true;
}

const int WINDOW_WIDTH = 640;
const int WINDOW_HEIGHT = 480;

void showMat(const string& title, Mat& img) {
    string wid = "Action: " + title;
    namedWindow(wid, 0);
    resizeWindow(wid, WINDOW_WIDTH, WINDOW_HEIGHT); 
    imshow(wid, img);  
}

typedef bool (*SeqFunc)(Mat& src, Mat& result, bool show);

// 颜色过滤
bool seq_colorFilter(Mat& src, Mat& result, bool show = true){
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    result.create(src.size(), src.type());
    for(int i=0;i<hsv.rows;i++){
        uchar* srcptr = src.ptr<uchar>(i);
        uchar* rowptr = hsv.ptr<uchar>(i);
        for(int j=0;j<hsv.cols;j++)
        {
            int H = int(rowptr[j*3 + 0]);  // 0-180
            int S = int(rowptr[j*3 + 1]);  // 0-255
            int V = int(rowptr[j*3 + 2]);  // 0-255

            if( ((H >= 0  && H <= 10) || (H >= 125 && H <= 180)) && S >= 43){
               result.ptr<uchar>(i)[j * 3 + 0] = src.ptr<uchar>(i)[j * 3 + 0];
               result.ptr<uchar>(i)[j * 3 + 1] = src.ptr<uchar>(i)[j * 3 + 1];
               result.ptr<uchar>(i)[j * 3 + 2] = src.ptr<uchar>(i)[j * 3 + 2];
            } else {
               result.ptr<uchar>(i)[j * 3 + 0] = 255;
               result.ptr<uchar>(i)[j * 3 + 1] = 255;
               result.ptr<uchar>(i)[j * 3 + 2] = 255;
            }
        }
    }
    hsv.release();
    if (show) {
        showMat("颜色过滤", result);  
    }
    return true;
}

// 移除高光
bool seq_highlightRemove(Mat& src, Mat& result, bool show = true){
    result.create(src.size(), src.type());
    
    for (int i = 0; i < src.rows; i++) {
        uchar* rowptr = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
				float B = rowptr[j * 3 + 0] / 255.0;
				float G = rowptr[j * 3 + 1] / 255.0;
				float R = rowptr[j * 3 + 2] / 255.0;
 
				float alpha_r = R / (R + G + B);
				float alpha_g = G / (R + G + B);
				float alpha_b = B / (R + G + B);
 
				float alpha = max(max(alpha_r, alpha_g), alpha_b);
				float MaxC = max(max(R, G), B);
				float minalpha = min(min(alpha_r, alpha_g), alpha_b);
				float beta_r = 1 - (alpha - alpha_r) / (3 * alpha - 1);
				float beta_g = 1 - (alpha - alpha_g) / (3 * alpha - 1);
				float beta_b = 1 - (alpha - alpha_b) / (3 * alpha - 1);
				float beta = max(max(beta_r, beta_g), beta_b);
				float gama_r = (alpha_r - minalpha) / (1 - 3 * minalpha);
				float gama_g = (alpha_g - minalpha) / (1 - 3 * minalpha);
				float gama_b = (alpha_b - minalpha) / (1 - 3 * minalpha);
				float gama = max(max(gama_r, gama_g), gama_b);
 
				float temp = (gama * (R + G + B) - MaxC) / (3 * gama - 1);
                     
				result.ptr<uchar>(i)[j * 3 + 0] = cvRound((B - (temp + 0.5)) * 255);
				result.ptr<uchar>(i)[j * 3 + 1] = cvRound((G - (temp + 0.5)) * 255);
				result.ptr<uchar>(i)[j * 3 + 2] = cvRound((R - (temp + 0.5)) * 255);
				
		}
	}

    if (show) {
        showMat("移除高光", result);  
    }

    return true;
}


// 灰度化
bool seq_gray(Mat& src, Mat& result, bool show = true){
    cvtColor(src, result, COLOR_BGR2GRAY);
    if (show) {
        showMat("灰度化", result);  
    }
    return true;
}

// 锐化
bool seq_sharpen(Mat& src, Mat& result, bool show = true){
    // Sharpen
    Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;    
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;

    result.create(src.size(),src.type());
    //对图像进行滤波
    filter2D(src, result, src.depth(), kernel);

    if (show) {
        showMat("锐化", result);  
    }
    return true;
}

// 高斯平滑
bool seq_guussian(Mat& src, Mat& result, bool show = true){
    GaussianBlur(src, result, Size(5,5), 0); 
    if (show) {
        showMat("高斯平滑", result);  
    }
    return true;
}

// 二值化
bool seq_binary(Mat& src, Mat& result, bool show = true){
    threshold(src, result, 100, 255, THRESH_BINARY_INV);
    if (show) {
        showMat("二值化", result);  
    }
    return true;
}

// 闭运算
bool seq_closure(Mat& src, Mat& result, bool show = true){
    Mat element = getStructuringElement(0, Size(5,5)); 
    morphologyEx(src, result, MORPH_CLOSE, element);
    if (show) {
        showMat("闭运算", result);  
    }
    return true;
}

// 腐蚀
bool seq_erode(Mat& src, Mat& result, bool show = true){
    Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
    erode(src, result, element);
    if (show) {
        showMat("腐蚀", result);  
    }
    return true;
}

// 边缘提取
bool seq_canny(Mat& src, Mat& result, bool show = true){
   
    if (show) {
        showMat("边缘提取", result);  
    }
    return true;
}

// 缩放
bool seq_resize(Mat& src, Mat& result, bool show = true){
    int srcSize = max(src.rows, src.cols);
    float scale = 160.0 / srcSize;

    int width = cvRound(scale * src.rows);
    int height = cvRound(scale * src.cols);

    result.create(width, height, CV_8UC3);
    resize(src, result, result.size());
        
    if (show) {
        showMat("缩放", result);  
    }
    return true;
}



int main(int argc, char *argv[])
{

    char *file = NULL;
    char *out = NULL;
  
    int opt = 0;
    while( (opt = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch (opt){
            case 'h':
            case '?': //如果是不能识别的选项，则opt返回'?'
                fprintf(stdout, "Usage: %s -f <filename> -o <output> [-h]\n", argv[0]);
                return 0;
            case 'f':
                file = optarg;
                break;
            case 'o':
                out = optarg;
                break;
        }
    }

    if (file == NULL || out == NULL) {
        fprintf(stdout, "Usage: %s -f <filename> -o <output> [-h]\n", argv[0]);
        return 0;
    }

    cout << "Built with OpenCV " << CV_VERSION << endl;

    map<string, SeqFunc> sequences;
    sequences.insert(make_pair("gray", seq_gray));
    sequences.insert(make_pair("sharpen", seq_sharpen));
    sequences.insert(make_pair("guussian", seq_guussian));
    sequences.insert(make_pair("binary", seq_binary));
    sequences.insert(make_pair("closure", seq_closure));
    sequences.insert(make_pair("erode", seq_erode));
    sequences.insert(make_pair("canny", seq_canny));
    sequences.insert(make_pair("highlightRemove", seq_highlightRemove));
    sequences.insert(make_pair("colorFilter", seq_colorFilter));
    sequences.insert(make_pair("resize", seq_resize));

    Mat img = imread(file);
    namedWindow("原图",0);
    resizeWindow("原图", 640, 480); 
    imshow("原图", img);

    Mat src = img.clone();
    //Mat gray;
    //seq_gray(img, gray, false);
    //cvtColor(gray, img, COLOR_GRAY2BGR);

    vector<string> pipeline;
    pipeline.push_back("colorFilter");
    //pipeline.push_back("highlightRemove");
    pipeline.push_back("gray");
    pipeline.push_back("closure");
    pipeline.push_back("binary");
    pipeline.push_back("erode");
    //pipeline.push_back("canny");
    
    //
    Mat result;
    for(vector<string>::iterator pit = pipeline.begin(); pit != pipeline.end(); ++pit) {
       map<string, SeqFunc>::iterator sit = sequences.find(*pit);
       if (sit == sequences.end()) {
           cerr << " can not found :" << *pit << endl; 
           continue;
       }
       sit->second(src, result, true);
       src = result.clone();
       result.release();
    }
   

    //轮廓检测
    vector<vector<Point> > contours;  //定义一个容器来存储所有检测到的轮廊
    vector<Vec4i> hierarchy;
    //轮廓检测函数
    Mat conv(src.size(), CV_8UC1);
    src.convertTo(conv, CV_8UC1);
    findContours(conv, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
    
    //取出区域
    vector<Rect> rects;
    char filename[25];
    int size = min(img.rows, img.cols);
    Rect numberRect = Rect(0, 0, 0, 0);
    vector<vector<Point> >::const_iterator itContours = contours.begin();
    long max_Index = static_cast<long>(contours.size());
    cout << "contours" << max_Index << endl;
    for (int i = 0; i <max_Index; ++i) {
        Rect rect = boundingRect(itContours[i]);
        numberRect = rect;

        // 剔除尺寸过小的区域(64*64)
        float rate = min(numberRect.width, numberRect.height) * 1.0 / max(numberRect.width, numberRect.height);
        cout << rate << numberRect.size() << " - hierarchy [" << i << "]: " << hierarchy[i][0] << "," << hierarchy[i][1] << "," << hierarchy[i][2] << endl;

        if ( rate < 0.6 || numberRect.height < 64 || numberRect.width < 64 ) {
            continue;
        }

        //Mat dst(numberRect.height, numberRect.width, CV_8UC4, numberRect.area());
        //dst = img(numberRect);
        //sprintf(filename, "%s/region-%d.jpg", out, i);
        //imwrite(filename, dst);
  
        if ( hierarchy[i][2] < 0 ) {
            continue;
        } 

        //cout << "hierarchy [" << i << "]: " << hierarchy[i][0] << "," << hierarchy[i][1] << "," << hierarchy[i][2] << endl;

        // 检查是否包含
        bool invalid = false;
        int num = rects.size();
        for(int j=0; j < num; j++) {
            Rect src = rects[j];
            Rect intersection = src & rect;
            if (intersection == rect){
                rects[j] = rect;
                invalid = true;
                break;
            }

            if (intersection == src) {
                invalid = true;
                break;
            }
        }

        if (invalid) {
            continue;
        }
        rects.push_back(numberRect);
        size = min(size, min(numberRect.width, numberRect.height));
    }

    // 根据x坐标排序
    int delta = size / 4;
    sort(rects.begin(),rects.end(),[&delta](const Rect &a, const Rect &b){
          return a.x < (b.x - delta) || a.y < (b.y - delta);  
    });

    createDir(out);
    vector<int> rows;
    vector<string> content;
    int num = rects.size();
    for(int i=0; i < num; i++) {
        numberRect = rects[i];
        Mat dst(numberRect.height, numberRect.width, CV_8UC4, numberRect.area());
        dst = img(numberRect);
        Mat dstResize; //我要转化为512*512大小的
        seq_resize(dst, dstResize, false);
        sprintf(filename, "%s/cell-%d.jpg", out, i);
        imwrite(filename, dstResize);

        // Create luminance  source
        Mat dstGray;
        cvtColor(dstResize, dstGray, COLOR_BGR2GRAY);
        sprintf(filename, "%s/cellgray-%d.jpg", out, i);
        imwrite(filename, dstGray);

        try {
            Ref<LuminanceSource> source = MatSource::create(dstGray);
            Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
            Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
            Ref<Reader> reader;
            reader.reset(new QRCodeReader);
            DecodeHints hints(DecodeHints::DEFAULT_HINT); 
		    hints.setTryHarder(true);   
            Ref<Result> result(reader->decode(bitmap, DecodeHints(DecodeHints::TRYHARDER_HINT)));

            // Get result point count
            int resultPointCount = result->getResultPoints()->size();
            if (resultPointCount > 0) {
                cout << "region-" << i << ":" << numberRect << endl;
                cout << "region-" << i << ":" << result->getText()->getText() << endl;
                
                // find row
                vector<int>::iterator it = find_if (rows.begin(), rows.end(), [&numberRect, &delta](int a){
                    return abs(numberRect.y - a) < delta;
                });

                if (it == rows.end()) {
                    rows.push_back(numberRect.y);
                    content.push_back(result->getText()->getText());
                } else {
                    int index = it - rows.begin();
                    content[index] +=  " " + result->getText()->getText();
                }
            }

        } catch (const ReaderException& e) {
            cerr << "ReaderException: " << e.what() << " (ignoring)" << endl;
        } catch (const zxing::IllegalArgumentException& e) {
            cerr << e.what() << " (ignoring)" << endl;
        } catch (const zxing::Exception& e) {
            cerr << e.what() << " (ignoring)" << endl;
        } catch (const std::exception& e) {
            cerr << e.what() << " (ignoring)" << endl;
        }
    }

    // Print
    num = content.size();
    cout << "Image Result: " << endl;
    for(int i=0; i < num; i++) {
        cout << content[i] << endl;
    }

    waitKey(0);
    return 0;
}
