

To preprocess and prepare all data run:

data_preperation.py --first_time_setup (you can omit this flag after the initial run)



How to do it manually (files are also called in data_preperation.py, and can 
be commented out depending on what operations you want done):

Order of operations:

1. run rename_files.py (only need to do this once)
2. normalize_images.py (only once per dataset)
3. big_geo_preproc.py
4. export.py
5. split.py
6. seperate.py