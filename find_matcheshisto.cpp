/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: find_matcheshisto.cpp
 *
 * This program finds and ranks the most similar images to a given target image based on 2D rg chromaticity histograms.
 * The feature vectors are computed using 2D rg chromaticity histograms, which capture the color distribution of the images.
 * The program reads precomputed feature vectors from a CSV file, computes the feature vector for the target image,
 * and then compares it to the feature vectors of other images to find the most similar ones.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Computes the 2D rg chromaticity histogram for the target image.
 * 3. Normalizes the histograms to ensure they are comparable across different images.
 * 4. Computes the distance between the target image's histogram and each image's histogram in the dataset.
 * 5. Sorts the images based on their computed distances in ascending order.
 * 6. Prints the top N most similar images.
 *
 * Functions:
 * - Mat computeRGChromaticityHistogram(const Mat& image, int bins):
 *     Computes the 2D rg chromaticity histogram for the given image.
 *
 * - Mat normalizeHistogram(const Mat& hist):
 *     Normalizes the given histogram so that the sum of its elements equals 1.
 *
 * - float computeHistogramIntersection(const Mat& hist1, const Mat& hist2):
 *     Computes the histogram intersection between two histograms.
 *
 * Usage:
 *   This program can be used to find and rank similar images in a dataset based on their 2D rg chromaticity histograms.
 *
 * Example:
 *   ./find_matcheshisto <target_image> <feature_file> <num_matches>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for finding and ranking similar images based on their 2D rg chromaticity histograms.
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

// Function to compute the histogram intersection
float computeHistogramIntersection(const Mat &hist1, const Mat &hist2)
{
    return sum(min(hist1, hist2))[0];
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <target_image> <feature_type> <feature_file> <num_matches>" << endl;
        return 1;
    }

    string targetImageFile = argv[1];
    string featureType = argv[2];
    string featureFile = argv[3];
    int numMatches = stoi(argv[4]);

    // Read the target image
    Mat targetImage = imread(targetImageFile);
    if (targetImage.empty())
    {
        cerr << "Error: Could not open target image " << targetImageFile << endl;
        return 1;
    }

    // Compute the feature vector for the target image based on the feature type
    Mat targetFeature;
    if (featureType == "rg_histogram")
    {
        // Compute the RG histogram feature vector
        targetFeature = computeRGChromaticityHistogram(targetImage, 16);
        targetFeature = normalizeHistogram(targetFeature);
    }
    else
    {
        cerr << "Error: Unknown feature type " << featureType << endl;
        return 1;
    }

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

        vector<float> featureVector;
        string value;
        while (getline(ss, value, ','))
        {
            featureVector.push_back(stof(value));
        }

        // Convert the feature vector to a Mat
        Mat featureMat(16, 16, CV_32F, featureVector.data());
        featureMat = normalizeHistogram(featureMat);

        // Compute the distance between the target image and the current image
        float distance = 1.0f - computeHistogramIntersection(targetFeature, featureMat);

        // Store the distance and filename
        distances.push_back(make_pair(distance, filename));
    }

    inFile.close();

    // Sort the distances in ascending order
    sort(distances.begin(), distances.end());

    // Print the top N matches
    cout << "Top " << numMatches << " matches:" << endl;
    for (int i = 0; i < numMatches && i < distances.size(); ++i)
    {
        cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
    }

    return 0;
}