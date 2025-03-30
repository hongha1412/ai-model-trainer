#!/bin/bash
# This script runs the AI server with optimized settings
# Used by Replit workflow

# Set environment variables to optimize performance
export DISABLE_AUTO_INSTALL=true
export FLASK_ENV=development
export SESSION_SECRET=dev-secret-key

# Run the server
./start_server.sh
