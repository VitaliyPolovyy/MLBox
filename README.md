--- DONE ---
- ML ops: Ray Serve: EPR=>Ray=>ITE (prototype requests)
- script utils: create "cvtools.py"
- script preprocessing: create script for converting raw images into clear dataset images ("a4_image_processor.py")
- dataset: take photos of all batches of peanuts
- dataset: rename the files according to the peanut batches
- dataset: a4 preprocessing of raw images
- dataset: auto annotation of 50 images
- dataset: manually annotation of 50 images: 90sec per image
- dataset 2: cut out peanuts individually and save them as separate images - test
- dataset: auto annotation of all images
- vast.ai: prepare script
- dataset: agree on an annotation with Svitlana
- vast.ai: train YOLOv8 on 50 pictures
- Git: create repo and commit, share to Andrii

--- IN PROGRESS ---
- RAY: create actual function in Ray (using pretrained yolo)
- ITE: save output image from Ray request into protocol in ERP

--- NEXT ---
- dataset: upload to CVAT all remaining images (split into parts of 200 images)
- dataset 2: cut out peanuts individually and save them as separate images (1200)
- dataset 2: annotate 1200 images for classification
- ML 2: train YOLO


sudo apt-get update
sudo apt-get install -y libgl1
pip install ultralytics
