/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: find_matchescolortexture.cpp
 *
 * Purpose:
 * This program finds and ranks the most similar images to a given target image based on combined color and texture feature vectors.
 * The feature vectors are computed using a combination of 3D RGB color histograms and texture histograms. The program reads precomputed
 * feature vectors from a CSV file, computes the feature vector for the target image, and then compares it to the feature vectors of other images
 * to find the most similar ones.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Computes the feature vector for the target image.
 * 3. Computes the distance between the target image's feature vector and each image's feature vector in the dataset.
 * 4. Sorts the images based on their computed distances in ascending order.
 * 5. Prints the top N most similar images.
 *
 * Functions:
 * - vector<pair<string, vector<float>>> readHistogramFeaturesFromFile(const string& featureFile):
 *     Reads histogram features from a CSV file.
 *
 * - float computeHistogramIntersection(const vector<float>& hist1, const vector<float>& hist2):
 *     Computes the histogram intersection between two histograms.
 *
 * - void findTopMatches(const string& targetImageFile, const vector<pair<string, vector<float>>>& features, int numMatches, int bins):
 *     Finds and prints the top N most similar images to the target image.
 *
 * Usage:
 *   This program can be used to find and rank similar images in a dataset based on their combined color and texture feature vectors.
 *
 * Example:
 *   ./find_matchescolortexture <target_image> <feature_file> <num_matches> <bins>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for finding and ranking similar images based on their combined color and texture feature vectors.
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

// Function to compute the RGB histogram for the whole image
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
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Compute histogram of gradient magnitudes
    int histSize = bins;
    float range[] = {0, 256};
    const float *histRange = {range};
    Mat hist;
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

// Function to compute the histogram intersection
float computeHistogramIntersection(const Mat &hist1, const Mat &hist2)
{
    return sum(min(hist1, hist2))[0];
}

// Function to read histogram features from a CSV file
vector<pair<string, vector<float>>> readHistogramFeaturesFromFile(const string &featureFile)
{
    vector<pair<string, vector<float>>> features;
    ifstream inFile(featureFile);
    if (!inFile)
    {
        cerr << "Error: Could not open feature file " << featureFile << endl;
        return features;
    }

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

        features.push_back(make_pair(filename, featureVector));
    }

    inFile.close();
    return features;
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
    Mat targetColorHist = normalizeHistogram(computeRGBHistogram(targetImage, bins));
    Mat targetTextureHist = normalizeHistogram(computeTextureHistogram(targetImage, bins));

    // Flatten the target histograms into a single feature vector
    vector<float> targetFeatureVector;
    for (int i = 0; i < bins; ++i)
    {
        for (int j = 0; j < bins; ++j)
        {
            for (int k = 0; k < bins; ++k)
            {
                targetFeatureVector.push_back(targetColorHist.at<float>(i, j, k));
            }
        }
    }
    for (int i = 0; i < bins; ++i)
    {
        targetFeatureVector.push_back(targetTextureHist.at<float>(i));
    }

    // Read the histogram features from the CSV file
    vector<pair<string, vector<float>>> features = readHistogramFeaturesFromFile(featureFile);

    // Vector to store the distances and corresponding filenames
    vector<pair<float, string>> distances;

    // Compute the distance between the target image and each image in the feature file
    for (const auto &feature : features)
    {
        const string &filename = feature.first;
        const vector<float> &featureVector = feature.second;

        // Split the feature vector into color and texture parts
        vector<float> colorFeatureVector(featureVector.begin(), featureVector.begin() + bins * bins * bins);
        vector<float> textureFeatureVector(featureVector.begin() + bins * bins * bins, featureVector.end());

        // Compute the histogram intersection for color and texture features
        float colorDistance = 0.0f;
        for (int i = 0; i < colorFeatureVector.size(); ++i)
        {
            colorDistance += min(targetFeatureVector[i], colorFeatureVector[i]);
        }

        float textureDistance = 0.0f;
        for (int i = 0; i < textureFeatureVector.size(); ++i)
        {
            textureDistance += min(targetFeatureVector[bins * bins * bins + i], textureFeatureVector[i]);
        }

        // Combine the distances (weighted average)
        float combinedDistance = (colorDistance + textureDistance) / 2.0f;

        // Store the distance and filename
        distances.push_back(make_pair(combinedDistance, filename));
    }

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