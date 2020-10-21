import json
import pandas as pd
import random

def get_annotations(metadata):
	r"""
	Args:
	metadata: Returns COCO data dictionary
	"""
	annotations_dict = {}
	# Gather all annotations associated with each image to a list, image id as key
	for annotation in metadata['annotations']:
		image_id = annotation['image_id']
		if image_id not in annotations_dict:
			annotations_dict[image_id] = []
			annotations_dict[image_id].append(annotation)
		else:
			#annotations_dict[image_id] = []
			if annotation not in annotations_dict[image_id]:
				annotations_dict[image_id].append(annotation)
			
	# Find out images without any annotations in database
	missing_annotation_count = 0
	for image in metadata['images']:
		image_id = image['id']
		if image_id not in annotations_dict:
			missing_annotation_count += 1
			annotations_dict[image_id] = []
	return annotations_dict, missing_annotation_count
	
def create_csv_files(output_folder):
	custom_labels = {'cobia'}
	label_map = {k: v + 1 for v, k in enumerate(custom_labels)}
	label_map['background'] = 0
	rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping

	data = json.load(open('master-coco/coco.json'))
	#annotations_dict, count = get_annotations(data)
	annotations_dict, _ = get_annotations(data)
	data = []
	keys = list(annotations_dict.keys())
	print ("Total images::", len(keys))
	
	#for key in keys:
	#	if len(annotations_dict[key])>=10:
	#		print (key)
	
	random.seed(6)
	random.shuffle(keys)
	
	test_images = keys[:400]
	train_images = keys[400:]
	train_data = []
	test_data = []
	
	#crowds = 0
	#test_crowds = 0
	for key in annotations_dict:
		if key in train_images:
			num_of_boxes = len(annotations_dict[key])
			for index in range(num_of_boxes):
				img_path = 'master-coco/' + annotations_dict[key][index]['image_id'] + '.jpg'
				label_id    = annotations_dict[key][index]['category_id']
				label    = rev_label_map[label_id]
				bbox     = annotations_dict[key][index]['bbox']
				x, y, w, h = bbox
				#print (x,y,w,h)
				xmax  = w+x
				ymax  = h+y
				value = (img_path,
						x,
						y,
						xmax,
						ymax,
						label)
				train_data.append(value)
				
		elif key in test_images:
			num_of_boxes = len(annotations_dict[key])
			for index in range(num_of_boxes):
				img_path = 'master-coco/' + annotations_dict[key][index]['image_id'] + '.jpg'
				label_id    = annotations_dict[key][index]['category_id']
				label    = rev_label_map[label_id]
				bbox     = annotations_dict[key][index]['bbox']
				x, y, w, h = bbox
				if key=='b72cc30b-9235-47a9-8bd8-8dc098091bfa':
					print (annotations_dict[key][index]['bbox'], num_of_boxes)
					break;
				#print (x,y,w,h)
				xmax  = w+x
				ymax  = h+y
				value = (img_path,
						x,
						y,
						xmax,
						ymax,
						label)
				test_data.append(value)
		
	#random.shuffle(data)
	#print (crowds, test_crowds)
	
	#test_data = data[:400]
	#train_data = data[400:]
	#if not os.path.exists(output_folder):
	#	os.makedirs(output_folder, exist_ok=True)
	
	train_df = pd.DataFrame(train_data)
	test_df = pd.DataFrame(test_data)
	print("Train data::",train_df.shape, " Test data::", test_df.shape)
	#print (train_df.head())
	train_df.to_csv((output_folder + '/train' + '_labels.csv'), index=None, header=None)
	test_df.to_csv((output_folder + '/test' + '_labels.csv'), index=None, header=None)
	
create_csv_files('dataset')
