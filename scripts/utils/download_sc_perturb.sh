#!/bin/bash

set -e

mkdir -p sc_data
cd sc_data

echo "Downloading Adamson..."
wget -c https://zenodo.org/record/7041849/files/adamson.h5ad

echo "Downloading Norman..."
wget -c https://zenodo.org/record/7041849/files/norman.h5ad

echo "Downloading Dixit..."
wget -c https://zenodo.org/record/7041849/files/dixit.h5ad

echo "Downloading sci-Plex3..."
wget -c https://zenodo.org/record/7041849/files/sciplex3.h5ad

echo "Download complete."
