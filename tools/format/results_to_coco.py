import json
from tqdm.auto import tqdm

with open('/mnt/c/Users/김동영/Downloads/AIFactory/ultralytics/runs/detect/base6_test/predictions.json', 'r') as f:
    json_data = json.load(f)
    
with open('/mnt/c/Users/김동영/Downloads/AIFactory/datasets/labels/answer_sample.json', 'r') as f:
    json_data_2 = json.load(f)
    
    
results = []
idx = 0
for i in tqdm(range(len(json_data))):
    for j in range(len(json_data_2['images'])):
        if json_data_2['images'][j]['file_name'] == (json_data[i]['image_id'] + '.png'):
            annotations = dict()
            annotations["id"] = idx
            annotations["image_id"] = json_data_2['images'][j]['id']
            annotations["category_id"] = json_data[i]['category_id']
            annotations["segmentation"] = []
            annotations["area"] = json_data[i]['bbox'][2] * json_data[i]['bbox'][3] 
            annotations["bbox"] = json_data[i]['bbox']
            annotations["iscrowd"] = 0
            annotations["attributes"] = {"occluded" : False, "rotation" : 0.0 }
            results.append(annotations)
            idx +=1
            break

with open('/mnt/c/Users/김동영/Downloads/AIFactory/datasets/labels/answer_sample.json', 'r') as f:
    json_data = json.load(f)
    json_data['annotations'] = results

with open('/mnt/c/Users/김동영/Downloads/AIFactory/dongyoung_prediction.json','w') as f1:
    json.dump(json_data,f1)
    
print('Done!')
