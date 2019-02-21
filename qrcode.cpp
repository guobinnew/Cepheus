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

//自定义“小于”
bool comp(const Rect &a, const Rect &b){
    Point atl = a.tl();
    Point btl = b.tl();
    return atl.x < btl.x || atl.y < btl.y;
}

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
    
    Mat img = imread(file);
    namedWindow("原图",0);
    resizeWindow("原图", 640, 480); 
    imshow("原图", img);

    //灰度化
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); 
    namedWindow("灰度化", 0);
    resizeWindow("灰度化", 640, 480); 
    imshow("灰度化", gray);

    // 高斯平滑
    Mat guussian;
    GaussianBlur(gray,guussian,Size(5,5),0); 
    namedWindow("高斯平滑", 0);
    resizeWindow("高斯平滑", 640, 480); 
    imshow("高斯平滑", guussian);

     //二值化
    Mat binary;
    threshold(guussian, binary, 100, 200, THRESH_BINARY);
    cvtColor(binary, img, COLOR_GRAY2BGRA);
    namedWindow("二值化", 0);
    resizeWindow("二值化", 640, 480); 
    imshow("二值化", binary);

    //闭运算
    Mat closure;
    Mat element = getStructuringElement(0,Size(7,7)); 
    morphologyEx(binary, closure, MORPH_CLOSE, element);   
    namedWindow("闭运算", 0);
    resizeWindow("闭运算", 640, 480); 
    imshow("闭运算", closure);

    //腐蚀
    Mat ercode;
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(13, 13));
    erode(closure, ercode, erodeElement);
    namedWindow("腐蚀", 0);
    resizeWindow("腐蚀", 640, 480);
    imshow("腐蚀", ercode);

    /*
    // 高斯平滑
    Mat guussian;
    GaussianBlur(ercode,guussian,Size(3,3),0); 
    namedWindow("高斯平滑", 0);
    resizeWindow("高斯平滑", 640, 480); 
    imshow("高斯平滑", guussian);

    // 边缘提取
    Mat edge;
    Canny(guussian, edge, 100, 200);
    namedWindow("边缘提取", 0);
    resizeWindow("边缘提取", 640, 480); 
    imshow("边缘提取", edge);
    */

    //轮廓检测
    vector<vector<Point> > contours;  //定义一个容器来存储所有检测到的轮廊
    vector<Vec4i> hierarchy;
    //轮廓检测函数
    Mat conv(ercode.size(), CV_8UC1);
    ercode.convertTo(conv, CV_8UC1);
    findContours(conv, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
    
    //取出身份证号码区域
    vector<Rect> rects;
    Rect numberRect = Rect(0, 0, 0, 0);
    vector<vector<Point> >::const_iterator itContours = contours.begin();
    long max_Index = static_cast<long>(contours.size());
    cout << "contours" << max_Index << endl;
    for (int i = 0; i <max_Index; ++i) {
        Rect rect = boundingRect(itContours[i]);
        numberRect = rect;

        // 剔除尺寸过小的区域(64*64)
        float rate = min(numberRect.width, numberRect.height) * 1.0 / max(numberRect.width, numberRect.height);
        if ( rate < 0.85 || numberRect.height < 64 || numberRect.width < 64 ) {
            continue;
        }
  
        if ( hierarchy[i][0] >= 0 || hierarchy[i][1] >= 0 || hierarchy[i][2] < 0) {
            continue;
        } 

        cout << "hierarchy [" << i << "]: " << hierarchy[i][0] << "," << hierarchy[i][1] << "," << hierarchy[i][2] << endl;

        // 检查是否包含
        bool invalid = false;
        int num = rects.size();
        for(int j=0; j < num; j++) {
            Rect src = rects[j];
            Rect intersection = src & rect;
            if (intersection == rect){
                invalid = true;
                break;
            }

            if (intersection == src) {
                rects[j] = rect;
                invalid = true;
                break;
            }
        }

        if (invalid) {
            continue;
        }
        rects.push_back(numberRect);
    }

    // 根据x坐标排序
    sort(rects.begin(),rects.end(),comp);

    createDir(out);
    vector<string> content;
    int num = rects.size();
    for(int i=0; i < num; i++) {
        numberRect = rects[i];
        Mat dst(numberRect.height, numberRect.width, CV_8UC4, numberRect.area());
        dst = img(numberRect);
        char filename[25];
        sprintf(filename, "%s/region-%d.jpg", out, i);
        
        imwrite(filename, dst);

        // Create luminance  source
        Mat dstGray;
        cvtColor(dst, dstGray, COLOR_BGR2GRAY);
        try {
            Ref<LuminanceSource> source = MatSource::create(dstGray);
            // Search for QR code
            Ref<Reader> reader;
            reader.reset(new QRCodeReader);
            Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
            Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
            Ref<Result> result(reader->decode(bitmap, DecodeHints(DecodeHints::TRYHARDER_HINT)));

            // Get result point count
            int resultPointCount = result->getResultPoints()->size();
            if (resultPointCount > 0) {
                cout << "region-" << i << ":" << result->getText()->getText() << endl;
                content.push_back(result->getText()->getText());
            }

        } catch (const ReaderException& e) {
            cerr << e.what() << " (ignoring)" << endl;
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
    cout << "Image Result: ";
    for(int i=0; i < num; i++) {
        cout << content[i] << " ";
    }
    cout << endl;

    waitKey(0);
    return 0;
}
