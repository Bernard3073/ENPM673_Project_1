#include <iostream>
// Include file for every supported OpenCV function
#include <opencv2/opencv.hpp>
// New C++ image display, sliders, buttons, mouse, I/O
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

RNG rng(12345);

bool compareContourArea(vector<Point> &contour1, vector<Point> &contour2){
    double a = fabs(contourArea(Mat(contour1)));
    double b = fabs(contourArea(Mat(contour2)));
    return a < b;
}

int main(int argc, char** argv) {
    
    VideoCapture cap ("/home/bernard/ENPM673/Project_1/Tag0.mp4");
    Mat testudo_img = imread("/home/bernard/ENPM673/Project_1/testudo.png");

    if (!cap.isOpened()) {
        cout << "Cannot find the video file !!!!" << endl;
        return -1;
    }
    
    while(1){
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        if(frame.empty()) break;
        double scale_percent = 60.0;
        double width = frame.cols*scale_percent/100.0; 
        double height = frame.rows*scale_percent/100.0;
        resize(frame, frame, Size(width, height), INTER_AREA);
        // Find contour
        Mat gray, blur, edges;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        bilateralFilter(gray, blur, 15, 75, 75);
        Canny(blur, edges, 50, 200);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        // Draw contours
        // Mat drawing = Mat::zeros(edges.size(), CV_8UC3 );
        // for( int i = 0; i< contours.size(); i++ )
        // {
        //     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //     drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        // }

        // ref: https://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea
        sort(contours.begin(), contours.end(), 
            [](const vector<Point>& c1, const vector<Point>& c2){
            return contourArea(c1, false) < contourArea(c2, false);});
        vector<vector<Point>> squares;

        // TODO: https://stackoverflow.com/questions/46187563/finding-largest-contours-c
        for(int i=0; i<contours.size(); i++){
            vector<Point> c = contours[i];
            double p = arcLength(c, true);
            vector<Point> approx;
            approxPolyDP(c, approx, 0.02*p, true);
            // if(approx.size()==4){
            //     if(contourArea(approx) > 20.0 && contourArea(approx) < 5000.0){
            //         squares.push_back(approx);
            //         cout << contourArea(approx) << endl; 
            //     }
            // }
            if(contourArea(approx) > 20.0 && contourArea(approx) < 5000.0){
                squares.push_back(approx);
                cout << contourArea(approx) << endl; 
            }
        }

        // Draw contours
        Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
        for( int i = 0; i< squares.size(); i++ )
        {
            // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            Scalar color = Scalar(0,0,255);
            drawContours( drawing, squares, i, color, 2, 8);
        }

        imshow( "Result window", drawing );
        // namedWindow("Simple Demo", cv::WINDOW_AUTOSIZE);
        // imshow("Simple Demo", frame);
        char c = (char) waitKey(25);
        if(c==27) break;
    }
    
    cap.release();
    destroyAllWindows();

    return 0;
}