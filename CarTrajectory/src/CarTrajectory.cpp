#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>

#define MIN_NUM_FEAT 2000

void TrackFeatures(cv::Mat aImage1, cv::Mat aImage2, std::vector<cv::Point2f>& aPoints1, std::vector<cv::Point2f>& aPoints2, std::vector<uchar>& aStatus) {

	// This function automatically gets rid of points for which tracking fails
	std::vector<float> err;

	// Optical Flow Algorithm Lucas-Kanade
	cv::calcOpticalFlowPyrLK(aImage1, aImage2, aPoints1, aPoints2, aStatus, err);

	// Getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	int indexCorrection = 0;

	cv::Rect roi(0, 0, aImage1.cols, aImage1.rows);

	size_t statusSize = aStatus.size();
	for (size_t i = 0; i < statusSize; ++i)
	{
		cv::Point2f pt = aPoints2.at(i - indexCorrection);
		if (aStatus.at(i) == 0 || !roi.contains(pt))
		{
			aPoints1.erase(aPoints1.begin() + (i - indexCorrection));
			aPoints2.erase(aPoints2.begin() + (i - indexCorrection));
			++indexCorrection;
		}
	}
}


void DetectFeatures(cv::Mat aImage, std::vector<cv::Point2f>& aPoints) //uses FAST as of now, modify parameters as necessary
{
	std::vector<cv::KeyPoint> keypoints;
	int fastThreshold = 20;
	bool nonmaxSuppression = true;
	// FAST features detection
	cv::FAST(aImage, keypoints, fastThreshold, nonmaxSuppression);
	cv::KeyPoint::convert(keypoints, aPoints, std::vector<int>());
}

int main(int argc, char** argv)
{
	int frameNumber;
	std::string dataDir;
	std::string calibFilePath;
	
	std::cout << "Please, Specify Path to Calibration Info!" << std::endl;
	std::cin >> calibFilePath;

	std::cout << "Please, Specify Data Directory!" << std::endl;
	std::cin >> dataDir;

	std::cout << "Please, Specify Frame Number!" << std::endl;
	std::cin >> frameNumber;

	cv::Mat image1;
	cv::Mat	image2;

	cv::Mat rotationMatrix;  //the final rotation and tranlation vectors containing the 
	cv::Mat translationVector;
	std::ofstream file1;
	file1.open("results.txt");

	double scale = 1.0;
	std::string fileName1 = dataDir + "000000.png";
	std::string fileName2 = dataDir + "000001.png";

	//read the first two frames from the dataset
	cv::Mat image1c = cv::imread(fileName1);
	cv::Mat image2c = cv::imread(fileName2);

	if (!image1c.data || !image2c.data)
	{
		std::cout << "(!) Error reading images. Uncorrect path or file name." << std::endl;
		return -1;
	}

	cv::cvtColor(image1c, image1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image2c, image2, cv::COLOR_BGR2GRAY);

	// Feature detection, tracking
	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2; // Vectors to store the coordinates of the feature points

	DetectFeatures(image1, points1); // Detect features in image1
	std::vector<uchar> status;
	TrackFeatures(image1, image2, points1, points2, status); // Track those features to image2

	std::ifstream file2;

	file2.open(calibFilePath.c_str());
	std::string str;
	std::vector<double> calib;

	while (std::getline(file2, str))
	{
		std::istringstream stream(str.substr(4));
		for (int i = 0; i < 4; ++i) {
			double value;
			stream >> value;
			calib.push_back(value);
		}
	}

	double focal = calib[0];
	const cv::Point2d pp(calib[2], calib[6]);

	// Recovering the pose and the essential matrix
	cv::Mat essentialMatrix;
	cv::Mat R;
	cv::Mat t;
	cv::Mat mask;

	essentialMatrix = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
	cv::recoverPose(essentialMatrix, points2, points1, R, t, focal, pp, mask);

	cv::Mat prevImage = image2;
	cv::Mat currImage;
	std::vector<cv::Point2f> prevFeatures = points2;
	std::vector<cv::Point2f> currFeatures;

	char filename[100];

	rotationMatrix = R.clone();
	translationVector = t.clone();

	cv::namedWindow("Pictures from Camera of Moving Car", cv::WINDOW_AUTOSIZE);

	file1 << "image1" << std::endl << "Rotation Matrix: " << std::endl << rotationMatrix << std::endl
		   << "Translation vector: " << std::endl << "[" << translationVector.at<double>(0) << " "
		   << translationVector.at<double>(1) << " " << translationVector.at<double>(2) << "]" << std::endl;

	for (int frameCount = 2; frameCount < frameNumber; ++frameCount)
	{
		std::string path = dataDir + "%06d.png";
		sprintf_s(filename, path.c_str(), frameCount);
		cv::Mat currImageC = cv::imread(filename);
		cv::cvtColor(currImageC, currImage, cv::COLOR_BGR2GRAY);
		std::vector<uchar> status;
		TrackFeatures(prevImage, currImage, prevFeatures, currFeatures, status);

		essentialMatrix = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
		cv::recoverPose(essentialMatrix, currFeatures, prevFeatures, R, t, focal, pp, mask);

		cv::Mat prevPts(2, (int)prevFeatures.size(), CV_64F);
		cv::Mat currPts(2, (int)currFeatures.size(), CV_64F);
		
		int prevFeaturesSize = (int)prevFeatures.size();
		for (int i = 0; i < prevFeaturesSize; ++i)
		{
			prevPts.at<double>(0, i) = prevFeatures.at(i).x;
			prevPts.at<double>(1, i) = prevFeatures.at(i).y;

			currPts.at<double>(0, i) = currFeatures.at(i).x;
			currPts.at<double>(1, i) = currFeatures.at(i).y;
		}

		if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
		{
			translationVector = translationVector + scale * (rotationMatrix * t);
			rotationMatrix = R * rotationMatrix;

		}

		file1 << "image" << frameCount << std::endl << "Rotation Matrix: " << std::endl << rotationMatrix << std::endl
			   << "Translation vector: " << std::endl << "[" << translationVector.at<double>(0) << " "
			   << translationVector.at<double>(1) << " " << translationVector.at<double>(2) << "]" << std::endl;

		if (prevFeatures.size() < MIN_NUM_FEAT)
		{
			DetectFeatures(prevImage, prevFeatures);
			TrackFeatures(prevImage, currImage, prevFeatures, currFeatures, status);
		}

		prevImage = currImage.clone();
		prevFeatures = currFeatures;

		cv::imshow("Pictures from Camera of Moving Car", currImageC);
		cv::waitKey(1);
	}

	return 0;
}