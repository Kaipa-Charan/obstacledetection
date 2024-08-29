#!/bin/bash

# Function to start server script on Raspberry Pi
start_server() {
    cd /home/raillabs/Desktop/jetsonboardobstacle/  # Replace with actual path
    python obstacledata.py &   # Replace with your server script name
    echo "Server script started."
}

# Function to send command to Jetson Orin and start client script
start_jetson_client() {
    # Replace with your Jetson Orin's IP address
    JETSON_IP='10.0.2.15'

    # Send command to activate conda environment and start client script
    ssh username@$JETSON_IP << EOF
    source /home/raillabs/archiconda3/envs/labelImg  # Replace with actual conda environment activation command
    cd /home/raillabs/labelImg/ultralytics/  # Replace with actual path
    python sharing_data.py &   # Replace with your client script name
    echo "Client script started on Jetson Orin."
EOF
}

# Main script starts here
echo "Choose an option:"
echo "1. Start server script on Raspberry Pi"
echo "2. Start client script on Nvidia Jetson Orin"

read choice

case $choice in
    1) start_server ;;
    2) start_jetson_client ;;
    *) echo "Invalid option. Exiting." ;;
esac

exit 0
