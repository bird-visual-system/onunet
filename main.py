from tensorflow.random import set_seed

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from pathlib import Path
from model import *
from data import *
 
from tensorflow.keras.utils import set_random_seed

#import json

# set random seed:
#set_seed(432)

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
set_random_seed(812)


# workaround to solve paths inconsistency on lenovomic and meteaub1804:
if os.path.isdir("/home/jovyan/kDrive/cellcount/cellcount"): # on meteaub1804
    mainpath = "/home/jovyan/kDrive/cellcount/cellcount/data/onunet"
    localpath = "/home/jovyan/onunet"
    path2weights = os.path.join(mainpath, "model_weights/optical_nerve_weights.hdf5")
    #path2history = os.path.join(mainpath, "model_weights/optical_nerve_training_history.csv")
    path2history = os.path.join(localpath, "optical_nerve_training_history.csv")
    path2setup = os.path.join(mainpath, "model_weights/optical_nerve_hyperparameters.txt")
else: # on lenovomic
    mainpath = "/home/jovyan/kDrive/cellcount/data/onunet"
    path2weights = os.path.join(mainpath, "model_weights/optical_nerve_weights_test.hdf5")
    path2history = os.path.join(mainpath, "model_weights/optical_nerve_training_history_test.csv")
    path2setup = os.path.join(mainpath, "model_weights/optical_nerve_hyperparameters_test.txt")

#path2pretrained_weights = os.path.join(mainpath, "model_weights", "27_09_2023", "optical_nerve_weights.hdf5")
#print(path2pretrained_weights)
path2data = os.path.join(mainpath, "optical_nerve")
path2train = os.path.join(path2data, "train")
print(path2train)
path2validation = os.path.join(path2data, "validation")
print(path2validation)

HP = dict(
    batch_size=int(os.getenv('BATCH_SIZE')),
    epochs=int(os.getenv('EPOCHS')),
    steps_per_epoch=int(os.getenv('STEPS_PER_EPOCH')),
    validation_steps=int(os.getenv('VALIDATION_STEPS'))
)

img_size = 512

# write hyperparameter setup to file:
print("Hyperparameters setup:")
with open(path2setup, 'w') as hp_file:
    #hp_file.write(json.dumps(HP))
    for k,v in HP.items():
        hp_line = f"{k} : {v}\n"
        print(hp_line)
        hp_file.write(hp_line)


data_gen_args = dict(rotation_range = 0.2,
                    width_shift_range = 0.05,
                    height_shift_range = 0.05,
                    shear_range = 0.05,
                    zoom_range = 0.05,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode = 'nearest')
augmented_data = os.path.join(path2train, "img_generator")

train_generator = trainGenerator(
    batch_size=HP['batch_size'],
    train_path=path2train,
    image_folder='images',
    mask_folder='labels',
    weight_folder='weights',
    aug_dict=data_gen_args,
    #save_to_dir=augmented_data,
    target_size=(img_size, img_size)
)

validation_generator = validationGenerator(
    batch_size=HP['batch_size'],
    validation_path=path2validation,
    image_folder='images',
    mask_folder='labels',
    weight_folder='weights',
    #aug_dict=data_gen_args,
    #save_to_dir=augmented_data,
    target_size=(img_size, img_size)
)

## sanity check:
num_batch = 3
for i, batch in enumerate(train_generator):
    inputs = batch[0]
    #targets = batch[1]
    image = inputs[0]
    print(f"image: min value is {np.min(image)} ; max value is {np.max(image)}")
    label = inputs[1]
    print(f"label: min value is {np.min(label)} ; max value is {np.max(label)}")
    weights = inputs[2]
    print(f"weights: min value is {np.min(weights)} ; max value is {np.max(weights)}")
    print('images batch shape :', image.shape)
    print('labels batch shape :', label.shape)
    print('weights batch shape :', weights.shape)

    if i == num_batch:
        break

# Define and train the model:
#model = unet(pretrained_weights=path2pretrained_weights)
#model = unet()
#model = unet_minimal()
model = weighted_loss_unet(input_shape=(img_size, img_size, 1))

csv_logger = CSVLogger(path2history, append=False)
model_checkpoint = ModelCheckpoint(path2weights, monitor='loss', verbose=1, save_best_only=True)
history = model.fit(
    x=train_generator,
    steps_per_epoch=HP['steps_per_epoch'], # number of batches
    epochs=HP['epochs'],
    validation_data=validation_generator,
    validation_steps=HP['validation_steps'],
    callbacks=[model_checkpoint, csv_logger])

