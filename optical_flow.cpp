//---------------------------------光流法对特定区域进行跟踪-----------------
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cstdio>

using namespace std;
using namespace cv;

//------------------------------------【全局函数声明】------------------------------
void tracking(Mat &frame,Mat &output);
//-------------------------------------【全局变量声明】-------------------------------
string window_name = "flow tracking";
Mat gray;                                                  //当前帧图片
Mat gray_prev;                                             //预测帧图片
Mat image;
vector<Point2f> points[2];                                 //point0为特征点的原来位置，point1位特征点新的位置
vector<uchar> status;                                      //跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;
Rect selection;
Point origin;                               //定义原点，
Point pointCentral;
bool selectObject = false;
int trackObject = 0;

static void onMouse(int event,int x,int y,int ,void*)
{
    if(selectObject)
    {
        selection.x=MIN(x,origin.x);
        selection.y=MIN(y,origin.y);
        selection.width = abs(x-origin.x);
        selection.height = abs(y-origin.y);

        selection &= Rect(0,0,image.cols,image.rows);     //保证selection在画面的里边
    }
    switch (event)
    {
        case EVENT_LBUTTONDOWN:
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            selectObject = true;
            break;
        case EVENT_LBUTTONUP:
            selectObject = false;
            if(selection.width > 0 && selection.height > 0)
            {
                trackObject = -1;
            }
            break;
    }
}


int main() {
    Mat frame;//定义视频帧
    Mat result;//定义结果
    VideoCapture capture(0);                       //读取视频
    if(capture.isOpened())                                 //如果视频打开成功，则进行以下步骤
    {
        while(true)
        {
            capture>>frame;                                //读取当前视频帧到frame中
            frame.copyTo(image);
            setMouseCallback(window_name,onMouse,0);
            if(!frame.empty())                             //如果当前帧不为空
            {
                tracking(image,result);                    //调用定义的函数，开始跟踪
            }
            else
            {
                cout<<"没有视频帧";                          //否则报错
                break;
            }
            int c = waitKey(50);                           //每隔50ms刷新一次
            if((char)c == 27)                              //用户在50ms以内按下“esc”这个按键，则跳出循环
            {
                break;
            }
            switch (c)
            {
                case 'c':                                  //停止追踪
                    trackObject = 0;
                    break;
                default:
                    break;
            }
        }
    }
    return 0;
}

void tracking(Mat &frame,Mat &output)
{
    cvtColor(frame,gray,COLOR_BGR2GRAY);                   //将当前的视频帧转换为灰度图，保存到gray中
    frame.copyTo(output);                                  //拷贝当前帧到输出output中
    if(selectObject)
    {
        rectangle(output,Point(selection.x,selection.y),Point(selection.x+selection.width,
                                                              selection.y+selection.height),Scalar(255,0,0));
    }
    //鼠标抬起时，进行检测
    if(trackObject == -1)
    {
        //选取selection区域的中心点为初始点
        pointCentral = Point(selection.x+selection.width/2,selection.y+selection.height/2);
        points[0].push_back(pointCentral);

        if(gray_prev.empty())
        {
            gray.copyTo(gray_prev);                            //如果前一帧为空，则复制当前帧到前一帧
        }
        if(points[0].size() == 0)
        {
            cout<<"这里错了；额"<<endl;
        }
        calcOpticalFlowPyrLK(gray_prev,gray, points[0],points[1],status,err);
        //绘制跟踪框图，以point【1】为中心，与selection的长和宽相同的矩形
        rectangle(output,Point(points[1][0].x - selection.width/2,points[1][0].y+selection.height/2),
                  Point(points[1][0].x + selection.width/2,points[1][0].y-selection.height/2),Scalar(255,0,0),3,8,0);
        //画线，在output图像上，点initial[i]到点points[1][i]的直线段，颜色为（0，0，255）
        line(output,points[0][0],points[1][0],Scalar(0,0,255),4,8);
        //画圆，在output图像上，圆心为point[1][i],半径为3，圆的颜色为（0，255，0）
        circle(output,points[1][0],6,Scalar(0,255,0),-1,8);

        swap(points[1],points[0]);                             //交换特征点容器points[0]和特征点容器points[1]
        swap(gray_prev,gray);                                  //把当前帧的图像赋值为上一帧图像，以便传入下一次迭代的calcOpticalFlowPyrLK
    }
    imshow(window_name,output);                            //在窗口window_name展示输出的结果
}
