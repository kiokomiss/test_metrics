#pragma once
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\photo.hpp"

#include "QualityMetrics.h"


void PsnrMetric::compute(const cv::Mat & img1, const cv::Mat & img2)
{
	mse->compute(img1, img2);
	result = 20 * log10(255) - 10 * log10(mse->getResult());
}

void MseMetric::compute(const cv::Mat& img1, const cv::Mat& img2)
{
	CV_Assert(img1.cols == img2.cols && img1.rows == img2.rows);
	CV_Assert(img1.channels() == img2.channels());
	CV_Assert(img1.channels() == 1 || img2.channels() == 3);

	switch (_colorspaceType)
	{
		case GRAY:
			result = computeMSEGrayscale(img1, img2);
			break;
		case RGB:
		{
			result = computeMSE_RGB(img1, img2);
		}
			break;
		case HSL:
		{
			result = computeMSE_HSL(img1, img2);
		}
		break;
		case YCbCr:
		{
			result = computeMSE_YCbCr(img1, img2);
		}
		break;
		default:
			std::cout << "Achtung!  Unknown color space type!" << std::endl;
			if (img1.channels() == 1)
			{
				std::cout << "Using GRAY type instead" << std::endl;
				result = computeMSEGrayscale(img1, img1);
			}
			else if (img1.channels() == 3)
			{
				std::cout << "Using RGB type instead" << std::endl;
				result = computeMSE_RGB(img1, img2);
			}
			break;
	}

}

double MseMetric::computeMSEGrayscale(const cv::Mat & img1, const cv::Mat & img2)
{
	CV_Assert(img1.channels() == 1 && img2.channels() == 1);
	double res_tmp = 0.0;
	int dif = 0;
	for (int y = 0; y < img1.rows; ++y)
	{
		const uchar* img1_row = img1.ptr<uchar>(y);
		const uchar* img2_row = img2.ptr<uchar>(y);

		for (int x = 0; x < img1.rows; ++x)
		{
			dif = (img1_row[x] - img2_row[x]);
			res_tmp += (double) (dif*dif);
		}
	}
	res_tmp /= (double)(img1.cols * img1.rows);
	return res_tmp;
}

double MseMetric::computeMSE_RGB(const cv::Mat & img1, const cv::Mat & img2)
{
	std::vector<cv::Mat> img1_channels;
	cv::split(img1, img1_channels);
	std::vector<cv::Mat> img2_channels;
	cv::split(img2, img2_channels);
	double result = 0.0;
	for (size_t i = 0; i < img1_channels.size(); ++i)
	{
		result += computeMSEGrayscale(img1_channels[i], img2_channels[i]);
	}
	return result /= (double)img1.channels();
}

double MseMetric::computeMSE_HSL(const cv::Mat & img1, const cv::Mat & img2)
{
	cv::Mat img1HSL;
	cv::cvtColor(img1, img1HSL, CV_BGR2HLS);
	cv::Mat img2HSL;
	cv::cvtColor(img2, img2HSL, CV_BGR2HLS);
	std::vector<cv::Mat> img1_channels;
	cv::split(img1HSL, img1_channels);
	std::vector<cv::Mat> img2_channels;
	cv::split(img2HSL, img2_channels);
	double result = 0.0;
	for (size_t i = 0; i < img1_channels.size(); ++i)
	{
		result += computeMSEGrayscale(img1_channels[i], img2_channels[i]);
	}
	return result /= (double)img1.channels();
}

double MseMetric::computeMSE_YCbCr(const cv::Mat & img1, const cv::Mat & img2)
{
	cv::Mat img1YCrCb;
	cv::cvtColor(img1, img1YCrCb, CV_BGR2YCrCb);
	cv::Mat img2YCrCb;
	cv::cvtColor(img2, img2YCrCb, CV_BGR2YCrCb);
	std::vector<cv::Mat> img1_channels;
	cv::split(img1YCrCb, img1_channels);
	std::vector<cv::Mat> img2_channels;
	cv::split(img2YCrCb, img2_channels);
	double  result = 0.0;
	for (size_t i = 0; i < img1_channels.size(); ++i)
	{
		result += computeMSEGrayscale(img1_channels[i], img2_channels[i]);
	}
	return result /= (double)img1.channels();
}

void SsimMetric::computeSSIM(const cv::Mat & img1, const cv::Mat & img2)
{
	CV_Assert(img1.cols == img2.cols && img1.rows == img2.rows);
	CV_Assert(img1.channels() == img2.channels());
	CV_Assert(img1.channels() == 1);

	const unsigned int	x_block_num = img1.cols / blockSize;
	const unsigned int y_block_num = img1.rows / blockSize;
	//CV_Assert(!x_block_num || !y_block_num);
	
	ssim_map.create(cv::Size(x_block_num * blockSize, y_block_num * blockSize), CV_32FC1);

	std::vector<double>	ssim_blocks;
	const double	n_samples = blockSize * blockSize;
	// Loop through all blocks
	for (unsigned int y_block = 0; y_block < y_block_num; ++y_block)
	{
		for (unsigned int x_block = 0; x_block < x_block_num; ++x_block)
		{
			double	ref_acc = 0.0, ref_acc_2 = 0.0,
				cmp_acc = 0.0, cmp_acc_2 = 0.0,
				ref_cmp_acc = 0.0,
				ref_avg = 0.0, ref_var = 0.0,
				cmp_avg = 0.0, cmp_var = 0.0,
				ref_cmp_cov = 0.0,
				ssim_num = 0.0, ssim_den = 0.0,
				ssim_block = 0.0;

			for (unsigned int j = 0; j < blockSize; ++j)
			{
				const uchar* img1_row = img1.ptr<uchar>(y_block + j);
				const uchar* img2_row = img2.ptr<uchar>(y_block + j);
				for (unsigned int i = 0; i < blockSize; ++i)
				{
					const unsigned char val1 = img1_row[x_block + i];
					const unsigned char val2 = img2_row[y_block + i];

					ref_acc += val1;
					ref_acc_2 += (val1*val1);
					cmp_acc += val2;
					cmp_acc_2 += (val2*val2);
					ref_cmp_acc += (val1*val2);
				}
			}
			// SSIM computation for a block
			ref_avg = ref_acc / n_samples,
				ref_var = ref_acc_2 / n_samples - (ref_avg*ref_avg),
				cmp_avg = cmp_acc / n_samples,
				cmp_var = cmp_acc_2 / n_samples - (cmp_avg*cmp_avg),
				ref_cmp_cov = ref_cmp_acc / n_samples - (ref_avg*cmp_avg),
				ssim_num = (2.0*ref_avg*cmp_avg + c1)*(2.0*ref_cmp_cov + c2),
				ssim_den = (ref_avg*ref_avg + cmp_avg*cmp_avg + c1)*(ref_var + cmp_var + c2),
				ssim_block = ssim_num / ssim_den;
			ssim_blocks.push_back(ssim_block);
			(cv::Mat(cv::Size(blockSize, blockSize), CV_32FC1, cv::Scalar::all(ssim_block))).copyTo(ssim_map(cv::Rect(y_block*blockSize, x_block* blockSize, blockSize, blockSize)));
		}
	}
	for (std::vector<double>::const_iterator it = ssim_blocks.begin(); it != ssim_blocks.end(); ++it)
	{
		ssim += *it;
	}
	ssim  /= ssim_blocks.size();
}

void SsimMetric::compute(const cv::Mat& ref, const cv::Mat& img)
{
	const int depth = CV_32F;

	cv::Mat I1, I2;
	ref.convertTo(I1, depth);
	img.convertTo(I2, depth);
	if (_colorspaceType == GRAY)
	{
		cvtColor(I1, I1, CV_BGR2GRAY);
		cvtColor(I2, I2, CV_BGR2GRAY);
	}
	cv::Mat I2_2 = I2.mul(I2); // I2^2
	cv::Mat I1_2 = I1.mul(I1); // I1^2
	cv::Mat I1_I2 = I1.mul(I2); // I1 * I2

	cv::Mat mu1, mu2;
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;

	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	cv::Mat t1, t2;
	cv::Mat numerator;
	cv::Mat denominator;

	// t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = 2 * mu1_mu2 + c1;
	t2 = 2 * sigma12 + c2;
	numerator = t1.mul(t2);

	// t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	t1 = mu1_2 + mu2_2 + c1;
	t2 = sigma1_2 + sigma2_2 + c2;
	denominator = t1.mul(t2);

	// ssim_map =  numerator./denominator;
	cv::divide(numerator, denominator, ssim_map);

	// mssim_avg = average of ssim map
	cv::Scalar mssim_avg = cv::mean(ssim_map);

	if (I1.channels() == 1)
		result =  mssim_avg[0];
	else	result = (mssim_avg[0] + mssim_avg[1] + mssim_avg[3]) / 3;
}