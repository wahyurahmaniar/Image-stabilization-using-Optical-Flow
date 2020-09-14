// Stabilization.cpp //

//#include "pch.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videostab.hpp"

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

//const int SMOOTHING_RADIUS = 15; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da) {
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	double da; // angle
};

struct Trajectory
{
	Trajectory() {}
	Trajectory(double _x, double _y, double _a) {
		x = _x;
		y = _y;
		a = _a;
	}
	// "+"
	friend Trajectory operator+(const Trajectory &c1, const Trajectory  &c2) {
		return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1, const Trajectory  &c2) {
		return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1, const Trajectory  &c2) {
		return Trajectory(c1.x*c2.x, c1.y*c2.y, c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1, const Trajectory  &c2) {
		return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx) {
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x, y, a);
	}

	double x;
	double y;
	double a; // angle
};
//
int main(int argc, char **argv)
{
	/*if(argc < 2) {
		cout << "./VideoStab [video.avi]" << endl;
		cin.get();
		return 0;
	}*/
	// For further analysis
	ofstream out_transform("prev_to_cur_transformation.txt");
	ofstream out_trajectory("trajectory.txt");
	ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	ofstream out_new_transform("new_prev_to_cur_transformation.txt");
	ofstream flow1("previous.txt");
	ofstream flow2("current.txt");

	const string source = "./actions.avi";

	VideoCapture cap(source);
	assert(cap.isOpened());

	Mat cur, cur_grey;
	Mat prev, prev_grey;
	Mat src1, src2;

	cap >> prev;//get the first frame.ch
	resize(prev, src1, Size(), 0.5, 0.5);
	cvtColor(src1, prev_grey, COLOR_BGR2GRAY);

	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransformParam> prev_to_cur_transform; // previous to current
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	// Step 2 - Accumulate the transformations to get the image trajectory
	vector <Trajectory> trajectory; // trajectory at all frames
	//
	// Step 3 - Smooth out the trajectory using an averaging window
	vector <Trajectory> smoothed_trajectory; // trajectory at all frames
	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd, pstd, pstd);// process noise covariance
	Trajectory R(cstd, cstd, cstd);// measurement noise covariance 
	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <TransformParam> new_prev_to_cur_transform;
	//
	// Step 5 - Apply the new transformation to the video
	//cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2, 3, CV_64F);

	int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
	//VideoWriter outputVideo; 
	//outputVideo.open("result_stab.avi" , CV_FOURCC('X','V','I','D'), 20, cvSize(cur.rows, cur.cols), true);  
   //

	int k = 1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	char name[100];
	Mat last_T;
	Mat prev_grey_, cur_grey_;
	
	time_t start, end;
	time(&start);

	while (true) {

		cap >> cur;
		if (cur.data == NULL) 
		{
			break;
		}

		resize(cur, src1, Size(), 0.5, 0.5);
		cvtColor(src1, cur_grey, COLOR_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <uchar> status;
		vector <float> err;

		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

		// weed out bad matches
		for (size_t i = 0; i < status.size(); i++) {
			if (status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}
		flow1 << prev_corner2 << endl;
		flow2 << cur_corner2 << endl;

		// translation + rotation only
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing
		//cout << "Transformation:" << T << endl;

		// in rare cases no transform is found. We'll just use the last known good transform.
		if (T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
		//
		//prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

		out_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		// Accumulated frame to frame transform
		x += dx;
		y += dy;
		a += da;
		//trajectory.push_back(Trajectory(x,y,a));
		//
		out_trajectory << k << " " << x << " " << y << " " << a << endl;
		//
		z = Trajectory(x, y, a);
		//
		if (k == 1) {
			// intial guesses
			X = Trajectory(0, 0, 0); //Initial estimate,  set 0
			P = Trajectory(1, 1, 1); //set error variance,set 1
		}
		else
		{
			//time updateprediction
			X_ = X; //X_(k) = X(k-1);
			P_ = P + Q; //P_(k) = P(k-1)+Q;
			// measurement update£¨correction£©
			K = P_ / (P_ + R); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_ + K * (z - X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
			P = (Trajectory(1, 1, 1) - K)*P_; //P(k) = (1-K(k))*P_(k);
		}
		//smoothed_trajectory.push_back(X);
		out_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
		//-
		// target - current
		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;

		//new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
		//
		out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);

		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

		Mat cur2;

		resize(prev, prev, cur.size());
		warpAffine(prev, cur2, T, cur.size());

		cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols - HORIZONTAL_BORDER_CROP));

		// Resize cur2 back to cur size, for better side by side comparison
		resize(cur2, cur2, cur.size());

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols * 2 + 3, cur.type());

		prev.copyTo(canvas(Range::all(), Range(0, cur.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur.cols + 3, cur.cols * 2 + 3)));

		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		if (canvas.cols > 1920) {
			resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
		}
		//outputVideo<<canvas;

		//sprintf(name, "./images_%d.jpg", k);
		//imwrite(name, cur2);
		imshow("Video Stabilization Result", canvas);


		waitKey(1);
		//
		prev = src1.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		//cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
		k++;

		if (k++ == 100)
		{
			time(&end);
			double seconds = difftime(end, start);
			cout << "Time taken : " << seconds << " seconds" << endl;
			double fps = 100 / seconds;
			cout << "Estimated frames per second : " << fps << endl;
		}

	}
	return 0;
}
