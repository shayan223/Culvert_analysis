#!/usr/bin/python3

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup-data", action="store_true", help="Setup data.")
    parser.add_argument("--first-time-setup", action="store_true", help="First time setting up data.")
    parser.add_argument("--rename", action="store_true", help="Rename data files.")
    parser.add_argument("--normalize", action="store_true", help="Noramlize image.")

    parser.add_argument("--train-model", action="store_true", help="Train model.")
    parser.add_argument("--backbone", choices=["detr_resnet101", "detr_resnet50", "detr_resnet34", "detr_resnet18"], default="detr_resnet101", help="DETR Backbone (DEFAULT: detr_resnet50).")

    parser.add_argument("--validate-model", action="store_true", help="Check accuracy of model.")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.setup_data:
        sys.path.insert(0, os.path.join(script_dir, "src/setup_data"))
        from main import main as setup_data_main
        setup_data_main(args)
    
    if args.train_model:
        from src.engine import main as engine_main
        engine_main(args.backbone)

    if args.validate_model:
        sys.path.insert(0, os.path.join(script_dir, "src"))
        from main import main as validation_main
        validation_main()


if __name__ == "__main__":
    main()