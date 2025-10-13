# Deployment Packager

## Component Overview
This component packages, compresses, and sends all deployments to the middleware. Based on the files created by the retrainer, a compressed ZIP file is created. Using the provided UUID of the target device within a deployment stream, the component ensures that the maximum message size of 8 MB is not exceeded and that all packages are delivered to the middleware.

## Prerequisites
This is a pyhton poetry project, this requires:
- Python >=3.11
- Python poetry (Packages Libaries in virtual env.)

## Operational flow
1. Store your deployment within the raw_data folder 
2. Execute the project
3. Folder will be zipped into zipped_data folder
4. Zip will be send to NATS Jetstream

## Execution
1. Pull directory to your local machine
2. Switch into the directory
3. Open terminal within the directory
4. Start a poetry virtual environment by typing "poetry shell"
5. Run the command "poetry run nats-model-service [deployment-folder name]"; Operational flow above will be executed.
