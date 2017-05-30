import os 
import shutil 

TRAIN_PER_CLASS = 60
VAL_PER_CLASS = 10
TEST_PER_CLASS = 10 
NUM_CLASS = 257 
 

src_dir = "U:\Downloads\\256_ObjectCategories"
train_dir = "U:\Downloads\data_train"
val_dir = "U:\Downloads\data_val"
test_dir = "U:\Downloads\data_test"

directories  = [train_dir , val_dir, test_dir]
for dirc in directories: 
	if not os.path.exists(dirc):
		os.makedirs(dirc)


for category in os.listdir(src_dir): 
	print 'Copying: ', category
	category_path = os.path.join(src_dir,category)
	
	train_path = os.path.join(train_dir,category)
	val_path = os.path.join(val_dir,category)
	test_path = os.path.join(test_dir,category)

	dest_paths = [train_path, val_path, test_path]
	
	for path in dest_paths: 
		if not os.path.exists(path):
			os.makedirs(path)

	image_paths = [os.path.join(category_path,image) for image in os.listdir(category_path)  
					if image.lower().endswith('.jpg') ]
	
	for train_image_path in image_paths[:TRAIN_PER_CLASS]:
		if not os.path.exists(os.path.join(train_path, os.path.basename(train_image_path))):
			shutil.copy2(train_image_path, train_path)
	for val_image_path in image_paths[TRAIN_PER_CLASS: TRAIN_PER_CLASS + VAL_PER_CLASS]:
		if not os.path.exists(os.path.join(val_path, os.path.basename(val_image_path))):
			shutil.copy2(val_image_path, val_path)
	for test_image_path in image_paths[-TEST_PER_CLASS:]:	
		if not os.path.exists(os.path.join(test_path, os.path.basename(test_image_path))):
			shutil.copy2(test_image_path, test_path)
		