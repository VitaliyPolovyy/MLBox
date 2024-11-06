import json
import os

from utils.cvtools import combine_coco_annotation_files

folder_path = r"C:\My storage\Python projects\DataSets\peanu    ts\task3"  # Replace with your folder path
annotation_data =combine_coco_annotation_files(os.path.join(folder_path, "result"))


output_path = os.path.join(folder_path, "annotation_coco.json")

# Write the JSON data to the specified file
with open(output_path, 'w') as f:
    json.dump(annotation_data, f)
