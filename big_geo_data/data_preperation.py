
from rename_files import rename_files
from normalize_images import normalize_images
from big_geo_preproc import big_geo_preproc
from export import export
from split import split_files
from seperate import seperate_files
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--rename", 
                    help="Calls rename function (only needs to be called once)",
                    action='store_true')

parser.add_argument("--normalize", 
                    help="Generates normalized images (only needs to be called once)",
                    action='store_true')

parser.add_argument("--first_time_setup",
                    help="Call this on first run",
                    action='store_true')
args = parser.parse_args()

if(args.rename or args.first_time_setup):
    print('############## Renaming Files ###############')
    rename_files()
if(args.normalize or args.first_time_setup):
    print(' ############## Normalizing Images ###############')
    normalize_images()
big_geo_preproc()
export()
split_files()
seperate_files()

