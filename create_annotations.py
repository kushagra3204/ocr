import os
import json
from PIL import Image

def convert_to_coco(base_dir, output_file):
    categories = []
    images = []
    annotations = []

    category_id = 1
    annotation_id = 1

    for subdir in os.listdir(os.path.join(base_dir, 'Train')):
        category = {'id': category_id, 'name': subdir}
        categories.append(category)
        category_id += 1

    for split in ['Train', 'Test', 'Validation']:
        data_dir = os.path.join(base_dir, split)
        for category in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, category)
            if not os.path.isdir(class_dir):
                continue

            for image_name in os.listdir(class_dir):
                if image_name.endswith(('.jpg', '.png')):
                    image_path = os.path.join(class_dir, image_name)
                    print(image_path)
                    image = Image.open(image_path)
                    width, height = image.size

                    image_info = {
                        'id': len(images) + 1,
                        'file_name': os.path.join(split, category, image_name),
                        'width': width,
                        'height': height
                    }
                    images.append(image_info)

                    bbox = [0, 0, width, height]
                    annotation = {
                        'id': annotation_id,
                        'image_id': image_info['id'],
                        'category_id': [cat['id'] for cat in categories if cat['name'] == category][0],
                        'bbox': bbox,
                        'area': width * height,
                        'iscrowd': 0
                    }
                    annotations.append(annotation)
                    annotation_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

base_dir = 'data'
output_file = 'annotations/coco_annotations.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
convert_to_coco(base_dir, output_file)