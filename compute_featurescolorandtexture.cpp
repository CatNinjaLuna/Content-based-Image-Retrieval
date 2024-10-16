/*
 * Author: Carolina Li
 * Date: Oct/13/2024
 * File: compute_featurescolorandtexture.cpp
 *
 * Purpose:
 * This program computes combined color and texture histograms for images in a specified directory
 * and writes the normalized histograms to a CSV file. The combined approach involves computing a
 * 3D RGB color histogram and a texture histogram using the Sobel operator for each image. This captures
 * both the color distribution and texture information of the image, which is useful for content-based
 * image retrieval (CBIR) systems.
 *
 * The program performs the following steps:
 * 1. Computes the 3D RGB color histogram for each image.
 * 2. Computes the texture histogram using the Sobel operator for each image.
 * 3. Normalizes the histograms to ensure they are comparable across different images.
 * 4. Writes the normalized histograms to a CSV file, with each row representing an image.
 *
 * Functions:
 * - Mat computeRGBHistogram(const Mat& image, int bins):
 *     Computes the 3D RGB color histogram for the given image.
 *
 * - Mat computeTextureHistogram(const Mat& image, int bins):
 *     Computes the texture histogram using the Sobel operator for the given image.
 *
 * - Mat normalizeHistogram(const Mat& hist):
 *     Normalizes the given histogram so that the sum of its elements equals 1.
 *
 * - void writeHistogramFeaturesToFile(const string& directory, const string& outputFile):
 *     Computes and writes the normalized histograms for all images in the specified directory to a CSV file.
 *
 * Usage:
 *   This program can be used to preprocess images for CBIR systems by extracting and storing their
 *   color and texture features in a compact and comparable form.
 *
 * Example:
 *   ./compute_featurescolorandtexture <image_directory> <output_file>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for feature extraction and storage.
 */

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Function to compute the RGB histogram for the entire image
Mat computeRGBHistogram(const Mat &image, int bins)
{
    int histSize[] = {bins, bins, bins};
    Mat hist(3, histSize, CV_32F, Scalar(0));

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            Vec3b pixel = image.at<Vec3b>(y, x);
            int rBin = min(static_cast<int>(pixel[2] * bins / 256), bins - 1);
            int gBin = min(static_cast<int>(pixel[1] * bins / 256), bins - 1);
            int bBin = min(static_cast<int>(pixel[0] * bins / 256), bins - 1);
            hist.at<float>(rBin, gBin, bBin)++;
        }
    }
    return hist;
}

// Function to compute the texture histogram using Sobel magnitude
Mat computeTextureHistogram(const Mat &image, int bins)
{
    Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    Mat hist;
    int histSize = bins;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&grad, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    return hist;
}

// Function to normalize a histogram
Mat normalizeHistogram(const Mat &hist)
{
    Mat histNorm;
    hist.copyTo(histNorm);
    histNorm /= sum(hist)[0];
    return histNorm;
}

// Function to write histogram features to a CSV file
void writeHistogramFeaturesToFile(const string &directory, const string &outputFile)
{
    ofstream outFile(outputFile);
    if (!outFile)
    {
        cerr << "Error: Could not open output file " << outputFile << endl;
        return;
    }

    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file())
        {
            string filename = entry.path().string();
            Mat image = imread(filename);
            if (image.empty())
            {
                cerr << "Error: Could not open image " << filename << endl;
                continue;
            }

            int bins = 8;
            Mat colorHist = normalizeHistogram(computeRGBHistogram(image, bins));
            Mat textureHist = normalizeHistogram(computeTextureHistogram(image, bins));

            outFile << filename;
            for (int i = 0; i < bins; ++i)
            {
                for (int j = 0; j < bins; ++j)
                {
                    for (int k = 0; k < bins; ++k)
                    {
                        outFile << "," << colorHist.at<float>(i, j, k);
                    }
                }
            }
            for (int i = 0; i < bins; ++i)
            {
                outFile << "," << textureHist.at<float>(i);
            }
            outFile << endl;
        }
    }

    outFile.close();
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <image_directory> <output_file>" << endl;
        return 1;
    }

    string imageDirectory = argv[1];
    string outputFile = argv[2];

    writeHistogramFeaturesToFile(imageDirectory, outputFile);

    return 0;
}