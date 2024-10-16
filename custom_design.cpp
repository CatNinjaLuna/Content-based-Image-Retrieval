/*
 * Author: Carolina Li
 * Date: Oct/14/2024
 * File: custom_design.cpp
 *
 * Purpose:
 * This program implements a content-based image retrieval (CBIR) system designed to find images
 * similar to a given target image based on custom-designed feature vectors. The feature vectors
 * combine color histograms, texture histograms, and deep network embeddings to capture the unique
 * characteristics of specific types of images, such as sunsets.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Extracts the feature vector for the target image.
 * 3. Computes the distance between the target image's feature vector and each image's feature vector
 *    in the dataset using a custom distance metric that combines color, texture, and deep features.
 * 4. Sorts the images based on their computed distances in ascending order.
 * 5. Prints the top N most similar images and some of the least similar images.
 *
 * Usage:
 *   custom_design <feature_file> <num_matches> <image_dir>
 *
 * Arguments:
 *   <feature_file> - Path to the CSV file containing precomputed feature vectors.
 *   <num_matches>  - Number of top matching images to display.
 *   <image_dir>    - Directory containing the images.
 *
 * Example:
 *   ./custom_design features.csv 5 ../images
 *
 * This program is designed to help understand the effectiveness of combining color histograms,
 * texture histograms, and deep network embeddings for specific types of images in a CBIR system.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Function to read feature vectors from a CSV file
vector<pair<string, vector<float>>> readFeatureVectorsFromFile(const string &featureFile)
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

// Function to compute the cosine distance between two vectors
float computeCosineDistance(const vector<float> &v1, const vector<float> &v2)
{
    float dotProduct = 0.0f;
    float normV1 = 0.0f;
    float normV2 = 0.0f;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }

    normV1 = sqrt(normV1);
    normV2 = sqrt(normV2);

    return 1.0f - (dotProduct / (normV1 * normV2));
}

// Function to compute the combined distance
float computeCombinedDistance(const Mat &colorHist1, const Mat &colorHist2, const Mat &textureHist1, const Mat &textureHist2, const vector<float> &deepFeature1, const vector<float> &deepFeature2)
{
    float colorDistance = 1.0f - computeHistogramIntersection(colorHist1, colorHist2);
    float textureDistance = 1.0f - computeHistogramIntersection(textureHist1, textureHist2);
    float deepDistance = computeCosineDistance(deepFeature1, deepFeature2);

    // Combine distances with weights
    float combinedDistance = 0.4f * colorDistance + 0.3f * textureDistance + 0.3f * deepDistance;
    return combinedDistance;
}

// Function to find and print the top matches for a given target image
void findTopMatches(const string &targetImageFile, const vector<pair<string, vector<float>>> &features, int numMatches, const string &imageDir)
{
    // Extract the base name of the target image file
    string targetImageBaseName = fs::path(targetImageFile).filename().string();

    // Read the target image
    Mat targetImage = imread(imageDir + "/" + targetImageBaseName);
    if (targetImage.empty())
    {
        cerr << "Error: Could not open target image " << targetImageBaseName << endl;
        return;
    }

    // Compute the feature vector for the target image
    int bins = 8;
    Mat targetColorHist = normalizeHistogram(computeRGBHistogram(targetImage, bins));
    Mat targetTextureHist = normalizeHistogram(computeTextureHistogram(targetImage, bins));

    // Find the deep feature vector for the target image
    vector<float> targetDeepFeatureVector;
    for (const auto &feature : features)
    {
        if (feature.first == targetImageBaseName)
        {
            targetDeepFeatureVector = feature.second;
            break;
        }
    }

    if (targetDeepFeatureVector.empty())
    {
        cerr << "Error: Could not find deep feature vector for target image " << targetImageBaseName << endl;
        return;
    }

    // Vector to store the distances and corresponding filenames
    vector<pair<float, string>> distances;

    // Compute the distance between the target image and each image in the feature file
    for (const auto &feature : features)
    {
        const string &filename = feature.first;
        const vector<float> &deepFeatureVector = feature.second;

        // Read the image
        Mat image = imread(imageDir + "/" + filename);
        if (image.empty())
        {
            cerr << "Error: Could not open image " << filename << endl;
            continue;
        }

        // Compute the feature vector for the image
        Mat colorHist = normalizeHistogram(computeRGBHistogram(image, bins));
        Mat textureHist = normalizeHistogram(computeTextureHistogram(image, bins));

        // Compute the combined distance
        float distance = computeCombinedDistance(targetColorHist, colorHist, targetTextureHist, textureHist, targetDeepFeatureVector, deepFeatureVector);

        // Store the distance and filename
        distances.push_back(make_pair(distance, filename));
    }

    // Sort the distances in ascending order (since lower distance means more similarity)
    sort(distances.begin(), distances.end());

    // Print the top N matches
    cout << "Top " << numMatches << " matches for " << targetImageBaseName << ":" << endl;
    for (int i = 0; i < numMatches && i < distances.size(); ++i)
    {
        cout << distances[i].second << " (Combined Distance: " << distances[i].first << ")" << endl;
    }

    // Print some of the least similar results
    cout << "Some of the least similar results for " << targetImageBaseName << ":" << endl;
    for (int i = distances.size() - 1; i >= distances.size() - numMatches && i >= 0; --i)
    {
        cout << distances[i].second << " (Combined Distance: " << distances[i].first << ")" << endl;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <feature_file> <num_matches> <image_dir>" << endl;
        return 1;
    }

    string featureFile = argv[1];
    int numMatches = stoi(argv[2]);
    string imageDir = argv[3];

    // Read the feature vectors from the CSV file
    vector<pair<string, vector<float>>> features = readFeatureVectorsFromFile(featureFile);

    // Find and print the top matches for the specified target images
    findTopMatches("pic.0893.jpg", features, numMatches, imageDir);
    findTopMatches("pic.0164.jpg", features, numMatches, imageDir);

    return 0;
}