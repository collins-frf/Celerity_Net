from data import *
import glob

# change often - global variables
epoch_no = 1000
cuts = 1
test_set_length = len(glob.glob('./data/test/real_test/*.png'))
test_size = cuts * test_set_length
uncertainty_runs = 2
val_size = 300
batch_size = 10
filters = 64
real_or_fake = 'fake'
snap = True
activation = 'sigmoid'
loss = "mean_absolute_error"
lr=.0001
optimizer = tf.keras.optimizers.Nadam(lr=lr, beta_1=.9, beta_2=.999, epsilon=1e-7, schedule_decay=.004)
#name = loss + str(lr) +"_"+ activation +"_snap_" + str(filters)
name = "mean_absolute_error0.0001sigmoidsnap+64"
logs_path = "./runs/" + name

### interpolation settings

#label size before up interp (5m bathy resolution, ~1m image resolution)
bathy_rows=102
bathy_cols=100
#fcn input size
img_rows=512
img_cols=500

#zero all image/label left of zeroline
zeroline = 0
downsample_zeroline = int(zeroline*(bathy_cols/img_cols))

#display transect for plot
transect = int(.7*bathy_rows)

#height of tiff image in cells and meters
tiff_height = 1075
tiff_width = 512
crosshore_distance_meters = 500
alongshore_distance_meters = 512
real_image_original_height = 1334
real_image_original_width = 334

#to get to 1:1 cell size
gen_image_resize_height = 361*5
gen_image_resize_width = 194*5
real_image_resize_height = 2000
real_image_resize_width = 500
real_bathy_resize_height = 2000
real_bathy_resize_width = 500
UAS_image_resize_height = 1000
UAS_image_resize_width = 500

#select column to grab square image from in 1500x500 image/label
north_bound = [np.random.randint(0, 1488) for i in range(3500)]

#shift by this much
duckgen_offset = 65
synthetic_offset = 165

#shift image by this much
real_image_offset = 0
UAS_cell_offset = 100
#shift label by this much
real_cell_offset = 0
UAS_bathy_offset = 80

test_id_offset = 10000
epoch_no = 1000

