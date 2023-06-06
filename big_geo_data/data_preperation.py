#!/usr/bin/env python3

from argparse import ArgumentParser

from big_geo_preproc import big_geo_preproc
from export import export
from normalize_images import normalize_images
from rename_files import rename_files
from seperate import seperate_files
from split import split_files


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-r",
        "--rename",
        help="Calls rename function (only needs to be called once)",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        help="Generates normalized images (only needs to be called once)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--first-time-setup",
        help="Call this on first run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.rename or args.first_time_setup:
        print("############## Renaming Files ###############")
        rename_files()
    if args.normalize or args.first_time_setup:
        print("############## Normalizing Images ###############")
        normalize_images()

    big_geo_preproc()
    export()
    split_files()
    seperate_files()


if __name__ == "__main__":
    main()