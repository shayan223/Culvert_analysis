
from inference import inference
from gen_val_labels import gen_val_labels
from analyse_val import analyse_val
from qualitative_validation import qualitative_validation

print("Running Inference")
inference()
print("Generating Validation Labels")
gen_val_labels()
print("Analysing Validation")
analyse_val()
print("Qualitative Validation")
qualitative_validation()
print("Done")