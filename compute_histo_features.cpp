/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: compute_histo_features.cpp
 *
 * Purpose:
 * This program computes 2D rg chromaticity histograms for images in a specified directory and writes
 * the normalized histograms to a CSV file. The rg chromaticity histogram captures the color distribution
 * of an image in the rg color space, which is useful for content-based image retrieval (CBIR) systems.
 *
 * The program performs the following steps:
 * 1. Computes the 2D rg chromaticity histogram for each image in the specified directory.
 * 2. Normalizes the histograms to ensure they are comparable across different images.
 * 3. Writes the normalized histograms to a CSV file, with each row representing an image.
 *
 * Functions:
 * - Mat computeRGChromaticityHistogram(const Mat& image, int bins):
 *     Computes the 2D rg chromaticity histogram for the given image.
 *
 * - Mat normalizeHistogram(const Mat& hist):
 *     Normalizes the given histogram so that the sum of its elements equals 1.
 *
 * - void writeHistogramFeaturesToFile(const string& directory, const string& outputFile):
 *     Computes and writes the normalized histograms for all images in the specified directory to a CSV file.
 *
 * Usage:
 *   This program can be used to preprocess images for CBIR systems by extracting and storing their
 *   color distribution features in a compact and comparable form.
 *
 * Example:
 *   ./compute_histo_features <image_directory> <output_file>
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

// Function to compute the 2D rg chromaticity histogram
Mat computeRGChromaticityHistogram(const Mat &image, int bins)
{
    Mat hist = Mat::zeros(bins, bins, CV_32F);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            Vec3b pixel = image.at<Vec3b>(y, x);
            float r = pixel[2] / 255.0;
            float g = pixel[1] / 255.0;
            int rBin = min(static_cast<int>(r * bins), bins - 1);
            int gBin = min(static_cast<int>(g * bins), bins - 1);
            hist.at<float>(rBin, gBin)++;
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

            int bins = 16;
            Mat hist = normalizeHistogram(computeRGChromaticityHistogram(image, bins));

            outFile << filename;
            for (int i = 0; i < bins; ++i)
            {
                for (int j = 0; j < bins; ++j)
                {
                    outFile << "," << hist.at<float>(i, j);
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