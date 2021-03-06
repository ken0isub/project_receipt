import os
from data_prep import rename_files, run_augmentation, img_prep, prep_cv2, read_images
from model_CNN import make_model_cnn
from model_ML import prep_ml_data, randomized_search, make_ml_model
from model_similarity import validate_similarity
from validation_CNN import validate_cnn
from validation_ML import validate_ml, validate_combined_ml
from visualize_predictions import model_summary

scan_data = './scan_data'
training_dir = './training'
validation_dir = './validation'
augmented_training_dir = './augmented_images'
models_dir = './models'
models_summary = './models/summary'
cv2_label_dir = './models/CV2'

directories = [scan_data, training_dir, validation_dir, augmented_training_dir, models_dir,
               models_summary, cv2_label_dir]

for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

print('preparing training files..')
rename_files(scan_data)
print('done\n')

print('scratching training images..')
run_augmentation(training_dir, augmented_training_dir)
print('done\n')

print('reading images as a dataset..')
X, y_out, y = read_images(augmented_training_dir)
print('done\n')

print('preparing cv2 template images..')
prep_cv2(training_dir, cv2_label_dir)
print('done\n')

# print('making a CNN model..')
# make_model_cnn(scratched_training_dir, 'model_CNN')
# print('done\n')
#
# print('validating the CNN model..')
# validate_cnn('model_CNN', validation_dir)
# print('done\n')

print('preparing data for ML..')
X_train, X_test, y_train, y_test = prep_ml_data(X, y_out)
print('done\n')

print('making ML models..')
make_ml_model(X_train, X_test, y_train, y_test)
print('done\n')

print('validating ML models..')
print(validate_ml('models/ML'))
print('done\n')

print('validating combined ML model..')
print(validate_combined_ml('models/ML'))
print('done\n')

print('validating similarity model..')
print(validate_similarity(validation_dir))
print('done\n')

model_summary()