#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"

int main(int argc, char* argv[])
{
	const char* keys =
	{
		"{ hog          |1                   | Use HOG features | }"
		"{ fixed_window |0                   | Use fixed window | }"
		"{ singlescale  |0                   | Use single or multiscale | }"
		"{ show         |1                   | Show result or silent run | }"
		"{ lab          |0                   | Use Lab color space with HOG | }"
		"{ gray         |0                   | Use gray color space (without HOG) and raw pixels | }"
		"{ images       |images.txt          | Path to the txt file with frames names | }"
		"{ region       |region.txt          | Path to the txt file with ground truth | }"
		"{ output       |output.txt          | Path to the txt file with results | }"
		"{ logs         |0                   | More debug logs into cout | }"
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
	bool debugLogs = parser.get<int>("logs") != 0;
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, debugLogs);

  	// Read groundtruth for the 1st frame
  	std::ifstream groundtruthFile;
  	groundtruthFile.open(parser.get<std::string>("region"));
  	
	auto GetRectFromFile = [&groundtruthFile]()
		{
			std::string regionLine;
			std::getline(groundtruthFile, regionLine);

			std::istringstream ss(regionLine);

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
			float xMin = std::min(x1, std::min(x2, std::min(x3, x4)));
			float yMin = std::min(y1, std::min(y2, std::min(y3, y4)));
			float width = std::max(x1, std::max(x2, std::max(x3, x4))) - xMin;
			float height = std::max(y1, std::max(y2, std::max(y3, y4))) - yMin;

			return cv::Rect(cvRound(xMin), cvRound(yMin), cvRound(width), cvRound(height));
		};
	
	// Read Images
	std::ifstream listFramesFile;
	std::filesystem::path listFramesPath = parser.get<std::string>("images");
	listFramesFile.open(listFramesPath.string());
	if (!listFramesFile.is_open())
	{
		std::cout << "File " << listFramesPath.string() << " not opened!" << std::endl;
		return 1;
	}

	// Write Results
	std::ofstream resultsFile;
	std::string resultsPath = parser.get<std::string>("output");
	resultsFile.open(resultsPath);
	if (!resultsFile.is_open())
	{
		std::cout << "File " << resultsPath << " not opened!" << std::endl;
		return 2;
	}

	// Frame counter
	int nFrames = 0;

	std::string frameName;
	cv::Mat frame;
	while (getline(listFramesFile, frameName))
	{
		// Read each frame from the list
		std::string pathToFrame = (listFramesPath.remove_filename() / frameName).string();
		frame = cv::imread(pathToFrame, cv::IMREAD_COLOR);
		if (frame.empty())
		{
			std::cout << "Frame " << pathToFrame << " is empty" << std::endl;
		}

		cv::Rect gtRect = GetRectFromFile();

		std::cout << "\n_______________________________________________________________________" << std::endl;
		std::cout << nFrames << ": Ground truth " << gtRect << " on frame " << frame.size() << std::endl;

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0)
		{
			tracker.init(gtRect, frame);
			resultsFile << gtRect.x << "," << gtRect.y << "," << gtRect.width << "," << gtRect.height << std::endl;
		}
		// Update
		else
		{
			float confidence = 0.f;
			cv::Rect result = tracker.update(frame, confidence);

			std::cout << "Detected rect " << result << ", conf = " << confidence << ", IoU = " << ((gtRect & result).area() / (float)(gtRect | result).area()) << std::endl;

			cv::rectangle(frame, result.tl(), result.br(), cv::Scalar(255, 0, 255), 1, 8);
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << std::endl;
		}

		cv::rectangle(frame, gtRect.tl(), gtRect.br(), cv::Scalar(0, 255, 0), 1, 8);

		nFrames++;

		if (!SILENT)
		{
			cv::imshow("Image", frame);
			cv::waitKey(1);
		}
	}

	return 0;
}
