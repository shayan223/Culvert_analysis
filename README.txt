
Un-zip samples inside the ./big_geo_data/data directory.

Refer to the linked tutorial for specific data input/compatibility requirements for the faster rcnn model

To Run:
	1. configure your training parameters in config.py
	2. go to ./big_geo_data and run data_preperation.py --first_time_setup (you can omit this flag after initial run)
	3. go back to root and run engine.py to train the model
	4. run validate.py

Results may be viewed in ./validation_qualitative and ./validation_results directories.

Feel free to contact me with any questions.



Initially model was built using the following tutorial:
https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

