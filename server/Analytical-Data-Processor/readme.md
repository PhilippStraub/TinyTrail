# Analytical Data Processor

## Component Overview
This component manages all analytical data received by clients and stores it in a file system. By subscribing to an analysis stream and handling all incoming topics (which correspond to the client UUIDs), the data is processed, extracted, and organized based on these UUIDs. Later, the retrainer component accesses the files to use the image data. Any processing errors are caught and logged, without stopping the component.

## Prerequisites
This is a pyhton poetry project, this requires:
- Python >=3.11
- Python poetry (Packages Libaries in virtual env.)

## Operational flow
1. Execute the project
2. Service will connect to NATS Jetstream, listening to the "ANALYSIS" Stream to the subject "analysis.logs"
3. Received messages contain the sender's device ID
4. Device ID folder for the device will be created in the folder "analytical-data-received" 
5. All messages received from that device are stored inside
! Pay attentention to the amount of data in these folders!


## Execution
1. Pull directory to your local machine
2. Switch into the directory
3. Open terminal within the directory
4. Start a poetry virtual environment by typing "poetry shell"
5. Run the command "poetry run nats-analytical-receiver"; Operational flow above will be executed.
