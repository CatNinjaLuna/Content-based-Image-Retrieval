# Name: Carolina Li

# OS: MacOS

# IDE: Visual Studio Code 1.94.2

# I completed the core part of the project without the extensions.

# Image Feature Extraction and Matching

This repository contains programs for extracting image features and finding similar images based on various feature extraction methods.

## Compilation

To compile the programs, you can use the following commands:

```sh
g++ -std=c++17 -o compute_features compute_features.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o compute_histo_features compute_histo_features.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o compute_featuresmulti compute_featuresmulti.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o compute_featurescolorandtexture compute_featurescolorandtexture.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o find_matches find_matches.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o find_matcheshisto find_matcheshisto.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o find_matchesmulti find_matchesmulti.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o findmatcheswithdeepnetwork findmatcheswithdeepnetwork.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -o custom_design custom_design.cpp `pkg-config --cflags --libs opencv4`

Usage
Compute Features
Baseline Features
./compute_features <image_directory> baseline <output_file>

2D rg Chromaticity Histogram Features
./compute_histo_features <image_directory> <output_file>

Multi-Region RGB Histogram Features

./compute_featuresmulti <image_directory> <output_file>

Combined Color and Texture Features

./compute_featurescolorandtexture <image_directory> <output_file>

Find Matches

Baseline Features

./find_matches <target_image> baseline <feature_file> <num_matches>

2D rg Chromaticity Histogram Features

./find_matcheshisto <target_image> <feature_file> <num_matches>

Multi-Region RGB Histogram Features

./find_matchesmulti <target_image> <feature_file> <num_matches>

Deep Network Features
./findmatcheswithdeepnetwork <target_image> <feature_file> <num_matches>

To apply custom image processing techniques to an image:


./custom_design <input_image> <output_image> <processing_type> <parameters>


Example
To compute baseline features for images in the images directory and save the features to features.csv:

./compute_features images baseline features.csv

To find the top 5 matches for target.jpg using baseline features stored in features.csv:

./find_matches ../images/target.jpg baseline features.csv 5

Notes
Ensure that the image directory contains valid image files.
The feature files should be in CSV format with each row representing an image and its feature vector.
The programs assume that the feature vectors are precomputed and stored in the specified CSV files.

```
