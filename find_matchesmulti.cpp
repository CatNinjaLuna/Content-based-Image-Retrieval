/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: find_matchesmulti.cpp
 *
 * Purpose:
 * This program finds and ranks the most similar images to a given target image based on multi-region RGB histograms.
 * The feature vectors are computed using RGB histograms for different regions of the images, capturing both color distribution
 * and spatial information. The program reads precomputed feature vectors from a CSV file, computes the feature vector for the target image,
 * and then compares it to the feature vectors of other images to find the most similar ones.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Computes the multi-region RGB histogram for the target image.
 * 3. Normalizes the histograms to ensure they are comparable across different images.
 * 4. Computes the distance between the target image's histogram and each image's histogram in the dataset.
 * 5. Sorts the images based on their computed distances in ascending order.
 * 6. Prints the top N most similar images.
 *
 * Functions:
 * - Mat computeRGBHistogram(const Mat& image, int bins, Rect region):
 *     Computes the RGB histogram for a given region of the image.
 *
 * - Mat normalizeHistogram(const Mat& hist):
 *     Normalizes the given histogram so that the sum of its elements equals 1.
 *
 * - float computeHistogramIntersection(const Mat& hist1, const Mat& hist2):
 *     Computes the histogram intersection between two histograms.
 *
 * Usage:
 *   This program can be used to find and rank similar images in a dataset based on their multi-region RGB histograms.
 *
 * Example:
 *   ./find_matchesmulti <target_image> <feature_file> <num_matches> <bins>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for finding and ranking similar images based on their multi-region RGB histograms.
 */

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

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

// Function to compute the histogram intersection
float computeHistogramIntersection(const Mat &hist1, const Mat &hist2)
{
    return sum(min(hist1, hist2))[0];
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <target_image> <feature_file> <num_matches>" << endl;
        return 1;
    }

    string targetImageFile = argv[1];
    string featureFile = argv[2];
    int numMatches = stoi(argv[3]);

    // Read the target image
    Mat targetImage = imread(targetImageFile);
    if (targetImage.empty())
    {
        cerr << "Error: Could not open target image " << targetImageFile << endl;
        return 1;
    }

    // Compute the feature vector for the target image
    int bins = 8;
    Rect topHalf(0, 0, targetImage.cols, targetImage.rows / 2);
    Rect bottomHalf(0, targetImage.rows / 2, targetImage.cols, targetImage.rows / 2);
    Mat topHist = normalizeHistogram(computeRGBHistogram(targetImage, bins, topHalf));
    Mat bottomHist = normalizeHistogram(computeRGBHistogram(targetImage, bins, bottomHalf));

    // Vector to store the distances and corresponding filenames
    vector<pair<float, string>> distances;

    // Open the feature file
    ifstream inFile(featureFile);
    if (!inFile)
    {
        cerr << "Error: Could not open feature file " << featureFile << endl;
        return 1;
    }

    // Read the feature vectors from the file
    string line;
    while (getline(inFile, line))
    {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ',');

        int histSize[] = {bins, bins, bins};
        Mat hist1 = Mat::zeros(3, histSize, CV_32F);
        Mat hist2 = Mat::zeros(3, histSize, CV_32F);
        string value;
        for (int i = 0; i < bins; ++i)
        {
            for (int j = 0; j < bins; ++j)
            {
                for (int k = 0; k < bins; ++k)
                {
                    getline(ss, value, ',');
                    hist1.at<float>(i, j, k) = stof(value);
                }
            }
        }
        for (int i = 0; i < bins; ++i)
        {
            for (int j = 0; j < bins; ++j)
            {
                for (int k = 0; k < bins; ++k)
                {
                    getline(ss, value, ',');
                    hist2.at<float>(i, j, k) = stof(value);
                }
            }
        }

        // Normalize the histograms
        hist1 = normalizeHistogram(hist1);
        hist2 = normalizeHistogram(hist2);

        // Compute the histogram intersection between the target image and the current image
        float distance1 = computeHistogramIntersection(topHist, hist1);
        float distance2 = computeHistogramIntersection(bottomHist, hist2);

        // Combine the distances (weighted average)
        float combinedDistance = (distance1 + distance2) / 2.0;

        // Store the distance and filename
        distances.push_back(make_pair(combinedDistance, filename));
    }

    inFile.close();

    // Sort the distances in descending order (since higher intersection means more similarity)
    sort(distances.begin(), distances.end(), greater<pair<float, string>>());

    // Print the top N matches
    cout << "Top " << numMatches << " matches:" << endl;
    for (int i = 0; i < numMatches && i < distances.size(); ++i)
    {
        cout << distances[i].second << " (Intersection: " << distances[i].first << ")" << endl;
    }

    return 0;
}