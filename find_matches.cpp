/*
 * Author: Carolina Li
 * Date: Oct/13/2024
 * File: find_matches.cpp
 *
 * Purpose:
 * This program finds and ranks the most similar images to a given target image based on feature vectors.
 * The feature vectors are computed using different methods, such as a 7x7 square region from the center
 * of the image. The program reads precomputed feature vectors from a CSV file, computes the feature vector
 * for the target image, and then compares it to the feature vectors of other images to find the most similar ones.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Computes the feature vector for the target image.
 * 3. Computes the distance between the target image's feature vector and each image's feature vector in the dataset.
 * 4. Sorts the images based on their computed distances in ascending order.
 * 5. Prints the top N most similar images.
 *
 * Functions:
 * - vector<float> computeBaselineFeatureVector(const Mat& image):
 *     Computes the 7x7 square feature vector from the center of the given image.
 *
 * - float computeSSD(const vector<float>& vec1, const vector<float>& vec2):
 *     Computes the sum-of-squared-difference (SSD) between two feature vectors.
 *
 * Usage:
 *   This program can be used to find and rank similar images in a dataset based on their feature vectors.
 *
 * Example:
 *   ./find_matches <target_image> <feature_type> <feature_file> <num_matches>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for finding and ranking similar images based on their feature vectors.
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

// Function to compute the sum-of-squared-difference (SSD) between two feature vectors
float computeSSD(const vector<float> &vec1, const vector<float> &vec2)
{
    float ssd = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        float diff = vec1[i] - vec2[i];
        ssd += diff * diff;
    }
    return ssd;
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
    vector<float> targetFeatureVector;
    Mat targetFeature;
    if (featureType == "baseline")
    {
        targetFeatureVector = computeBaselineFeatureVector(targetImage);
    }
    else
    {
        cerr << "Error: Unknown feature type " << featureType << endl;
        return 1;
    }

    // Store the distances and corresponding filenames
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

        // Compute the distance between the target image and the current image
        float distance = computeSSD(targetFeatureVector, featureVector);

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