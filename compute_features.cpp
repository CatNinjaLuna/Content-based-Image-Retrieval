/*
 * Author: Carolina Li
 * Date: Oct/13/2024
 # File compute_features.cpp
 *
 * Purpose:
 * This program computes feature vectors for images in a specified directory and writes the
 * normalized feature vectors to a CSV file. The feature vectors are based on a 7x7 square
 * region from the center of the images, capturing the color distribution of the central part
 * of the images. This is useful for content-based image retrieval (CBIR) systems.
 *
 * The program performs the following steps:
 * 1. Reads the image filenames from the specified directory.
 * 2. Computes the 7x7 square feature vector from the center of each image.
 * 3. Writes the feature vectors to a CSV file, with each row representing an image.
 *
 * Functions:
 * - vector<float> computeBaselineFeatureVector(const Mat& image):
 *     Computes the 7x7 square feature vector from the center of the given image.
 *
 * - vector<string> readImageFilenames(const string& directory):
 *     Reads image filenames from the specified directory.
 *
 * Usage:
 *   This program can be used to preprocess images for CBIR systems by extracting and storing their
 *   central color distribution features in a compact and comparable form.
 *
 * Example:
 *   ./compute_features <image_directory> <feature_type> <output_file>
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

// Function to compute the 7x7 square feature vector from the center of the image
vector<float> computeBaselineFeatureVector(const Mat &image)
{
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int halfSize = 3;

    vector<float> featureVector;
    for (int y = centerY - halfSize; y <= centerY + halfSize; ++y)
    {
        for (int x = centerX - halfSize; x <= centerX + halfSize; ++x)
        {
            Vec3b pixel = image.at<Vec3b>(y, x);
            featureVector.push_back(pixel[0]);
            featureVector.push_back(pixel[1]);
            featureVector.push_back(pixel[2]);
        }
    }
    return featureVector;
}

// Function to read images from a directory and return their filenames
vector<string> readImageFilenames(const string &directory)
{
    vector<string> filenames;
    for (const auto &entry : fs::directory_iterator(directory))
    {
        filenames.push_back(entry.path().string());
    }
    return filenames;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <image_directory> <feature_type> <output_file>" << endl;
        return 1;
    }

    string imageDirectory = argv[1];
    string featureType = argv[2];
    string outputFile = argv[3];

    // Read the image filenames from the directory
    vector<string> imageFilenames = readImageFilenames(imageDirectory);

    ofstream outFile(outputFile);
    if (!outFile)
    {
        cerr << "Error: Could not open output file " << outputFile << endl;
        return 1;
    }

    // Loop over the images in the directory
    for (const string &filename : imageFilenames)
    {
        Mat image = imread(filename);
        if (image.empty())
        {
            cerr << "Error: Could not open image " << filename << endl;
            continue;
        }

        // Compute the feature vector for the current image based on the feature type
        vector<float> featureVector;
        if (featureType == "baseline")
        {
            featureVector = computeBaselineFeatureVector(image);
        }
        else
        {
            cerr << "Error: Unknown feature type " << featureType << endl;
            return 1;
        }

        // Write the filename and feature vector to the output file
        outFile << filename;
        for (float value : featureVector)
        {
            outFile << "," << value;
        }
        outFile << endl;
    }

    outFile.close();
    return 0;
}