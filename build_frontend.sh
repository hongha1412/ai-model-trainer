#!/bin/bash

echo "Building Vue.js frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js to build the frontend."
    exit 1
fi

# Navigate to the frontend directory
cd frontend

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the project
echo "Building project..."
npm run build

# Create static directory if it doesn't exist
mkdir -p ../static/vue

# Copy the dist folder to the static directory
echo "Copying build files to static/vue directory..."
cp -r dist/* ../static/vue/

echo "Build completed successfully."
echo "The Vue.js app is now available in the static/vue directory."