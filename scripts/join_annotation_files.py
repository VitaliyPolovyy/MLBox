import json
import os

from mlbox.utils.cvtools import combine_coco_annotation_files

folder_path = "/fmnt/c/My storage/Python projects/DataSets/peanuts/task_1_3_4_7"
annotation_data = combine_coco_annotation_files(folder_path)

output_path = os.path.join(folder_path, "annotation_coco.json")

# Write the JSON data to the specified file
with open(output_path, "w") as f:
    json.dump(annotation_data, f)
