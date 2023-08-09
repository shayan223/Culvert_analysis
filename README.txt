
Un-zip samples inside the ./big_geo_data/data directory.

Refer to the linked tutorial for specific data input/compatibility requirements for the faster rcnn model

This main branch tests the Faster-RCNN model

To Run:
	1. Unpack your images and xlsx format labels in the /big_geo_data/data directory
		*NOTE: Script tools for modifying your dataset to match the varying model requirements
		are in the dataset_utils directory. Feel free to use at your discretion.
	2. configure your training parameters in config.py
	3. go to ./big_geo_data and run data_preperation.py --first_time_setup (you can omit this flag after initial run)
		*NOTE: In case running this results in an error, omit the --first_time_setup flag, and it will pickup where it left off.
	4. go back to root and run engine.py to train the model
	5. run validate.py

Results may be viewed in ./validation_qualitative and ./validation_results directories.

Different models can be accessed via git branches for each.


Feel free to contact me with any questions.



Initially the faster rcnn model was built using the following tutorial:
https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

