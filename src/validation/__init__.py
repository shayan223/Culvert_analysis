from .analyse_val import analyse_val
from .gen_val_labels import gen_val_labels
from .inference import inference
from .qualitative_validation import qualitative_validation


def main(args):
    inference(args)
    # gen_val_labels()
    # analyse_val()
    # qualitative_validation()