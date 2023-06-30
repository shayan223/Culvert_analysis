# DETR Implementation Branch

## Data Setup

1. Install all the required libraries via: `pip3 install -r ./requirements.txt`.
2. Un-zip samples inside the `./data/` directory.
3. To setup the data, run the following command: `./main.py --setup-data
   --first-time-setup`.

## Configuration

All the configuration is available in the `src/config.py` file.

## Train the Model

To train the model, run: `./main.py --train-model`. You can also pass the
`--backbone` option with the following: `detr_resnet101`, `detr_resnet50`,
`detr_resnet34`, `detr_resnet18`.

## Validation

To check the accuracy of the model, run: `./main.py --validate-model`

## All at Once

If you don't want to run each individual command, run: `./main.py --all`.

## Directory Structure Explained

1. The `./data/` directory is where all the data's stored.
2. The `./src/setup_data/` directory stores the code to setup the data.
3. The `./src/validation/` directory stores the code to check the accuracy of
   the model.
4. The `./src/utils/` directory stores a ton of useful utility functions used
   throughout the codebase.
