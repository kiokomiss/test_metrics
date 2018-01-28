#ifndef QUALITY_METRICS_H
#define QUALITY_METRICS_H

#include <memory>

#include "opencv2/opencv_modules.hpp"

// TODO: Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) no-reference image quality score

typedef enum  MetricOptions { SSIM, PSNR, MSE } MetricOptions;

class QualityMetrics
{
public:
	enum ColorSpace { GRAY, HSL, RGB, YCbCr };

	QualityMetrics() {}
	virtual ~QualityMetrics() {}

	virtual void compute(const cv::Mat& img1, const cv::Mat & img2) = 0;
	virtual double getResult() = 0;
};

/* @brief Compute mean square error
	Assumes input images are 8bit 1-channel grayscale or 3-channel BGR images
*/
class MseMetric : public QualityMetrics
{
public:
	MseMetric(ColorSpace colorspaceType) : _colorspaceType(colorspaceType) {};
	~MseMetric() {};

	void compute(const cv::Mat& img1, const cv::Mat& img2);
	double getResult() { return result; };
private:
	double computeMSEGrayscale(const cv::Mat & img1, const cv::Mat & img2);
	double computeMSE_RGB(const cv::Mat & img1, const cv::Mat & img2);
	double computeMSE_HSL(const cv::Mat & img1, const cv::Mat & img2);
	double computeMSE_YCbCr(const cv::Mat & img1, const cv::Mat & img2);
private:
	ColorSpace _colorspaceType;
	double result = 0.0;
};

class PsnrMetric : public QualityMetrics
{
public:

	PsnrMetric(ColorSpace colorspaceType): mse(new MseMetric(colorspaceType)){}
	~PsnrMetric() {}
	void compute(const cv::Mat & img1, const cv::Mat & img2);
	double getResult() { return result; };
private:
	MseMetric* mse;
	double result = 0.0;

};

/* Computes SSIM map and average SSIM for 8-bit grayscale images 
*/
class SsimMetric : public QualityMetrics
{
public:
	SsimMetric(ColorSpace colorspaceType): _colorspaceType(colorspaceType){}\
	~SsimMetric() {}
	void computeSSIM(const cv::Mat & ref, const cv::Mat & img);
	//Computes MSSIM
	void compute(const cv::Mat & ref, const cv::Mat & img);
	double getSSIM() { return ssim; };
	double getMSSIM() { return result; };
	double getResult() { return result; };
	cv::Mat getSsimMap() { return ssim_map; };

private:
	ColorSpace _colorspaceType;
	float k1 = 0.01f;
	float k2 = 0.03f;
	//Dynamic range as 2^(number of bits per pixel) - 1
	int L = 255;
	float c1 = (k1*L)*(k1*L);
	float c2 = (k2*L)*(k2*L);
	//Standard block size = 8
	int blockSize = 8;
	//Gaussian kernel parameters
	int window_size = 11;
	float sigma = 1.5;
	cv::Mat ssim_map;
	//Average of ssim of all blocks 
	double ssim = 0.0;
	double result = 0.0;
};

static inline 
uchar sqr(uchar a)
{
	return a * a;
}

class MetricComputationFabric {
public:
	static std::shared_ptr<QualityMetrics> create(MetricOptions metricType, QualityMetrics::ColorSpace colorSpaceType) {
		switch (metricType)
		{
		case MSE:
			return std::shared_ptr<MseMetric>(new MseMetric(colorSpaceType));
			break;
		case PSNR:
			return std::shared_ptr<PsnrMetric>(new PsnrMetric(colorSpaceType));
			break;
		case SSIM:
			return std::shared_ptr<SsimMetric>(new SsimMetric(colorSpaceType));
			break;
		default:
		{
			std::cout << "F(create(metricType, colorSpaceType) Unsupported metricType. /n Using SSIM metric for grayscale images instead." << std::endl;
			return std::shared_ptr<SsimMetric>(new SsimMetric(colorSpaceType));
		}
		}
	}
};

#endif // QUALITY_METRICS_H