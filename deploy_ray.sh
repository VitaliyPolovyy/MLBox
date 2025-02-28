#!/bin/bash

clear

echo "Shutting down existing Ray Serve deployments..."
serve shutdown -y >/dev/null

MAX_RETRIES=10
RETRY_COUNT=0

# Wait for applications to shut down
while ! serve status | grep -q "applications: {}"; do
    echo "Applications are still shutting down. Retrying in 2 seconds..."
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))

    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Applications are still active after $MAX_RETRIES retries. Exiting."
        exit 1
    fi
done

echo "Deploying new configuration..."
serve deploy deployments/ray_serv.yaml >/dev/null

# Check deployment status in a loop
RETRY_COUNT=0
while true; do
    STATUS=$(serve status | grep -m1 -o "status: [A-Z_]*" | awk '{print $2}')
    
    if [ "$STATUS" == "RUNNING" ]; then
        echo "Deployment is successful!"
        break
    elif [ "$STATUS" == "DEPLOY_FAILED" ]; then
        echo "ERROR: Deployment has failed."
        exit 1
    elif [ "$STATUS" == "DEPLOYING" ]; then
        echo "Deployment is in progress..."
        sleep 1
    else
        echo "ERROR: Unexpected status: $STATUS"
        exit 1
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Deployment did not finish after $MAX_RETRIES retries. Exiting."
        exit 1
    fi
done
