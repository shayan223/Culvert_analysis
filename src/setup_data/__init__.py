from .big_geo_preproc import big_geo_preproc
from .export import export
from .normalize_images import normalize_images
from .rename_files import rename_files
from .seperate import seperate_files
from .split import split_files


def main(args):
    # if args.rename or args.first_time_setup:
    #     print("############## Renaming Files ###############")
    #     rename_files()
    # if args.normalize or args.first_time_setup:
    #     print("############## Normalizing Images ###############")
    #     normalize_images()

    # big_geo_preproc()
    # export()
    # split_files()
    seperate_files()