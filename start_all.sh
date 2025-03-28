#!/bin/bash

# Start both the backend server and frontend dev server
echo "Starting the API server and frontend..."

# Start the API server in the background
bash start_server.sh &
SERVER_PID=$!

# Wait a bit for the server to initialize
sleep 3

# Start the frontend
bash start_frontend.sh &
FRONTEND_PID=$!

# Function to handle exit and kill child processes
cleanup() {
    echo "Shutting down services..."
    kill $SERVER_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register the cleanup function for when script is interrupted
trap cleanup SIGINT SIGTERM

echo "Services started with PIDs: Server=$SERVER_PID, Frontend=$FRONTEND_PID"
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait