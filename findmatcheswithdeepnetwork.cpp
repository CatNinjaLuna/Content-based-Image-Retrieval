/*
 * Name: Carolina Li
 * Date: Oct/14/2024
 * File: findmatcheswithdeepnetwork.cpp
 *
 * This program finds and ranks the most similar images to a given target image based on deep network feature vectors.
 * The feature vectors are computed using a pre-trained deep neural network, capturing high-level features of the images.
 * The program reads precomputed feature vectors from a CSV file, computes the feature vector for the target image,
 * and then compares it to the feature vectors of other images to find the most similar ones.
 *
 * The program performs the following steps:
 * 1. Reads precomputed feature vectors from a CSV file.
 * 2. Computes the feature vector for the target image using a deep neural network.
 * 3. Computes the cosine distance between the target image's feature vector and each image's feature vector in the dataset.
 * 4. Sorts the images based on their computed distances in ascending order.
 * 5. Prints the top N most similar images.
 *
 * Functions:
 * - vector<pair<string, vector<float>>> readFeatureVectorsFromFile(const string& featureFile):
 *     Reads feature vectors from a CSV file.
 *
 * - vector<float> computeDeepNetworkFeatureVector(const Mat& image):
 *     Computes the feature vector for the given image using a pre-trained deep neural network.
 *
 * - float computeCosineDistance(const vector<float>& vec1, const vector<float>& vec2):
 *     Computes the cosine distance between two feature vectors.
 *
 * Usage:
 *   This program can be used to find and rank similar images in a dataset based on their deep network feature vectors.
 *
 * Example:
 *   ./findmatcheswithdeepnetwork <target_image> <feature_file> <num_matches>
 *
 * This file is designed to support the development of CBIR systems by providing essential utilities
 * for finding and ranking similar images based on their deep network feature vectors.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

using namespace std;
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

    // Extract the base name of the target image file
    string targetImageBaseName = fs::path(targetImageFile).filename().string();

    // Read the feature vectors from the CSV file
    vector<pair<string, vector<float>>> features = readFeatureVectorsFromFile(featureFile);

    // Debug: Print the target image base filename
    cout << "Target image base filename: " << targetImageBaseName << endl;

    // Find the feature vector for the target image
    vector<float> targetFeatureVector;
    for (const auto &feature : features)
    {
        // Debug: Print the current filename being checked
        // cout << "Checking filename: " << feature.first << endl;

        if (feature.first == targetImageBaseName)
        {
            targetFeatureVector = feature.second;
            break;
        }
    }

    if (targetFeatureVector.empty())
    {
        cerr << "Error: Could not find feature vector for target image " << targetImageBaseName << endl;
        return 1;
    }

    // Vector to store the distances and corresponding filenames
    vector<pair<float, string>> distances;

    // Compute the distance between the target image and each image in the feature file
    for (const auto &feature : features)
    {
        const string &filename = feature.first;
        const vector<float> &featureVector = feature.second;

        // Compute the cosine distance
        float distance = computeCosineDistance(targetFeatureVector, featureVector);

        // Store the distance and filename
        distances.push_back(make_pair(distance, filename));
    }

    // Sort the distances in ascending order (since lower distance means more similarity)
    sort(distances.begin(), distances.end());

    // Print the top N matches
    cout << "Top " << numMatches << " matches for " << targetImageBaseName << ":" << endl;
    for (int i = 0; i < numMatches && i < distances.size(); ++i)
    {
        cout << distances[i].second << " (Cosine Distance: " << distances[i].first << ")" << endl;
    }

    return 0;
}