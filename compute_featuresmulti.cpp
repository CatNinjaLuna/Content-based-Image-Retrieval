/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: compute_featuresmulti.cpp
 *
 * Purpose:
 * This program computes multi-histogram features for images in a specified directory and writes
 * the normalized histograms to a CSV file. The multi-histogram approach involves dividing the image
 * into multiple regions and computing a color histogram for each region. This captures both the color
 * distribution and spatial information of the image, which is useful for content-based image retrieval (CBIR) systems.
 *
 * The program performs the following steps:
 * 1. Divides each image into a grid of regions.
 * 2. Computes the color histogram for each region.
 * 3. Normalizes the histograms to ensure they are comparable across different images.
 * 4. Writes the normalized histograms to a CSV file, with each row representing an image.
 *
 * Functions:
 * - Mat computeRegionHistogram(const Mat& image, int bins, int gridX, int gridY):
 *     Computes the color histogram for a specific region of the given image.
 *
 * - Mat normalizeHistogram(const Mat& hist):
 *     Normalizes the given histogram so that the sum of its elements equals 1.
 *
 * - void writeHistogramFeaturesToFile(const string& directory, const string& outputFile):
 *     Computes and writes the normalized histograms for all images in the specified directory to a CSV file.
 *
 * Usage:
 *   This program can be used to preprocess images for CBIR systems by extracting and storing their
 *   color distribution and spatial features in a compact and comparable form.
 *
 * Example:
 *   ./compute_featuresmulti <image_directory> <output_file>
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

// Function to compute the RGB histogram for a given region of the image
Mat computeRGBHistogram(const Mat &image, int bins, Rect region)
{
    int histSize[] = {bins, bins, bins};
    Mat hist(3, histSize, CV_32F, Scalar(0));

    for (int y = region.y; y < region.y + region.height; ++y)
    {
        for (int x = region.x; x < region.x + region.width; ++x)
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
            Rect topHalf(0, 0, image.cols, image.rows / 2);
            Rect bottomHalf(0, image.rows / 2, image.cols, image.rows / 2);
            Mat topHist = normalizeHistogram(computeRGBHistogram(image, bins, topHalf));
            Mat bottomHist = normalizeHistogram(computeRGBHistogram(image, bins, bottomHalf));

            outFile << filename;
            for (int i = 0; i < bins; ++i)
            {
                for (int j = 0; j < bins; ++j)
                {
                    for (int k = 0; k < bins; ++k)
                    {
                        outFile << "," << topHist.at<float>(i, j, k);
                    }
                }
            }
            for (int i = 0; i < bins; ++i)
            {
                for (int j = 0; j < bins; ++j)
                {
                    for (int k = 0; k < bins; ++k)
                    {
                        outFile << "," << bottomHist.at<float>(i, j, k);
                    }
                }
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