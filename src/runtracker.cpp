#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"

int main(int argc, char* argv[])
{
	const char* keys =
	{
		"{ hog          |1                   | number of test case: 0 - Test KCF | }"
		"{ fixed_window |0                   | Left coordinate of the object | }"
		"{ singlescale  |0                   | Top coordinate of the object | }"
		"{ show         |1                   | Width of the bounding box | }"
		"{ lab          |0                   | Height of the bounding box | }"
		"{ gray         |0                   | Path to the folder with results | }"
		"{ images       |images.txt          | Path to the folder with results | }"
		"{ region       |region.txt          | Path to the folder with results | }"
		"{ output       |output.txt          | Path to the folder with results | }"
	};
	cv::CommandLineParser parser(argc, argv, keys);
	parser.printMessage();

	bool HOG = parser.get<int>("hog") != 0;
	bool FIXEDWINDOW = parser.get<int>("fixed_window") != 0;
	bool MULTISCALE = parser.get<int>("singlescale") == 0;
	bool SILENT = parser.get<int>("show") == 0;
	bool LAB = parser.get<int>("lab") != 0;
	if (LAB)
	{
		HOG = true;
		std::cout << "Lab is true: hog == " << HOG << std::endl;
	}
	if (parser.get<int>("gray") != 0)
	{
		HOG = false;
		LAB = false;
		std::cout << "gray is true: hog == " << HOG << ", LAB == " << LAB << std::endl;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

  	// Read groundtruth for the 1st frame
  	std::ifstream groundtruthFile;
	std::string groundtruth = parser.get<std::string>("region");
  	groundtruthFile.open(groundtruth);
  	std::string firstLine;
  	std::getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
	std::istringstream ss(firstLine);

  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 

	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  std::min(x1, std::min(x2, std::min(x3, x4)));
	float yMin =  std::min(y1, std::min(y2, std::min(y3, y4)));
	float width = std::max(x1, std::max(x2, std::max(x3, x4))) - xMin;
	float height = std::max(y1, std::max(y2, std::max(y3, y4))) - yMin;
	
	// Read Images
	std::ifstream listFramesFile;
	std::string listFrames = parser.get<std::string>("images");
	listFramesFile.open(listFrames);

	// Write Results
	std::ofstream resultsFile;
	std::string resultsPath = parser.get<std::string>("output");
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;

	std::string frameName;
	cv::Mat frame;
	while (getline(listFramesFile, frameName))
	{
		// Read each frame from the list
		frame = cv::imread(frameName, cv::IMREAD_COLOR);

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0)
		{
			tracker.init(cv::Rect(xMin, yMin, width, height), frame );
			rectangle( frame, cv::Point( xMin, yMin ), cv::Point( xMin+width, yMin+height), cv::Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << std::endl;
		}
		// Update
		else
		{
			cv::Rect result = tracker.update(frame);
			rectangle( frame, cv::Point( result.x, result.y ), cv::Point( result.x+result.width, result.y+result.height), cv::Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << std::endl;
		}

		nFrames++;

		if (!SILENT)
		{
			cv::imshow("Image", frame);
			cv::waitKey(1);
		}
	}

	return 0;
}
