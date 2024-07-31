#!/bin/bash

# Define the base directory
BASE_DIR="./"

# Create the base directory
mkdir -p $BASE_DIR

# Create the main package directory
mkdir -p $BASE_DIR/seaducks

# Create the main package files
touch $BASE_DIR/seaducks/__init__.py
touch $BASE_DIR/seaducks/data_processing.py
touch $BASE_DIR/seaducks/filtering.py
touch $BASE_DIR/seaducks/utils.py
touch $BASE_DIR/seaducks/config.py

# Create the tests directory and files
mkdir -p $BASE_DIR/tests
touch $BASE_DIR/tests/__init__.py
touch $BASE_DIR/tests/test_data_processing.py
touch $BASE_DIR/tests/test_filtering.py
touch $BASE_DIR/tests/test_utils.py

# Create the logs directory and a log file
mkdir -p $BASE_DIR/logs
touch $BASE_DIR/logs/data_processing.log

# Create the data directory
mkdir -p $BASE_DIR/data

# Create the scripts directory and main script
mkdir -p $BASE_DIR/scripts
touch $BASE_DIR/scripts/run_processing.py

# Create the setup.py file
touch $BASE_DIR/setup.py

# Create the README.md file
touch $BASE_DIR/README.md

# Create the requirements.txt file
touch $BASE_DIR/requirements.txt

echo "Directory structure for SeaDucks project created successfully."
