#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>


namespace fs = std::experimental::filesystem;

#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "QualityMetrics.h"

struct InputArguments {
	std::string dir1;
	std::string dir2;
	std::string outdir;
	MetricOptions metric = SSIM;
	QualityMetrics::ColorSpace colorSpace = QualityMetrics::ColorSpace::GRAY;
};

void parseInputArgs(int argc, char* argv[], InputArguments& inArgs)
{
	inArgs.dir1 = argv[1];
	inArgs.dir2 = argv[2];
	inArgs.outdir = argv[3];
	if (argc > 5) 
	{
		std::string metric = argv[4];
		if (metric == "SSIM") inArgs.metric = SSIM;
		else if (metric == "PSNR") inArgs.metric = PSNR;
		else if (metric == "MSE") inArgs.metric = MSE;
		else {
			std::cout << "Using default metric SSIM" << std::endl;
		}
	}
	if (argc > 6)
	{ 
		std::string colorspace = argv[5];
		if (colorspace == "Gray") inArgs.colorSpace = QualityMetrics::ColorSpace::GRAY;
		else if (colorspace == "RGB") inArgs.colorSpace = QualityMetrics::ColorSpace::RGB;
		else if (colorspace == "HSL") inArgs.colorSpace = QualityMetrics::ColorSpace::HSL;
		else if (colorspace == "YCbCr") inArgs.colorSpace = QualityMetrics::ColorSpace::YCbCr;
		else {
			std::cout << "Using default color space Gray" << std::endl;
		}
	}
}

void write_to_log(const std::string &msg, const std::string & logfilename)
{
	std::ofstream log_file(
		logfilename, std::ios_base::out | std::ios_base::app);
	log_file << msg << std::endl;
}

void help()
{
	std::cout << "Usage:	metric_computation.exe  <dir1> <dir2> <out> \n"
		<< " dir1 - full path of input directory with reference files \n"
		<< " dir2 - full path of input directory with files to compare \n"
		<< " out  - full path to directory for results \n";
}

int main(int argc, char* argv[])
{
	if (argc < 4) 
	{
		help();
		return -1;
	}

	InputArguments* inArgs = new InputArguments;
	parseInputArgs(argc, argv, *inArgs);

	std::string path1 = (*inArgs).dir1;
	std::string path2 = (*inArgs).dir2;
	std::vector<std::string> files_list1;
	std::vector<std::string> files_list2;
	//std::string ext(".bmp");

	for (auto & file1 : fs::directory_iterator(path1)) 
	{
		//	if (file.path().extension() == ext())
				fs::path file2 = path2;
				file2 /= file1.path().filename();
				std::cout << file2 << '\n';
				if (fs::exists(file2)) 
				{
					files_list1.push_back(file1.path().filename().string());
					files_list2.push_back(file2.filename().string());
					std::cout << file1 << '\n';
				}
	}
	
	std::string logfilename = (*inArgs).outdir + "\\" + "log.txt";

	for (int i = 0; i < files_list1.size(); ++i)
	{
		cv::Mat image1, image2;
		image1 = cv::imread(path1+ "\\" + files_list1[i]);
		image2 = cv::imread(path2 + "\\" + files_list2[i]);

		std::shared_ptr<QualityMetrics> metricComputation = MetricComputationFabric::create((*inArgs).metric, (*inArgs).colorSpace);

		metricComputation->compute(image1, image2);
		double mssim = metricComputation->getResult();
		write_to_log( files_list1[i] + "  mssim = " + std::to_string(mssim), logfilename);
		if ((*inArgs).metric == SSIM)
		{
			SsimMetric* ssim_comp = dynamic_cast<SsimMetric*>(metricComputation.get());
			cv::Mat ssim_map = ssim_comp->getSsimMap();
			ssim_map.convertTo(ssim_map, CV_8U, 255.0f);
			imwrite((*inArgs).outdir + "\\" + std::to_string(mssim) + "_" + files_list1[i], ssim_map);
		}
	}
	return 0;
}
