import numpy as np
import keras

import os
import shutil
import sys

#from docopt import docopt
#import envoy
from PIL import Image
#from PIL import ImageDraw

from matplotlib import pyplot as plt

from model import build_model, FRAME_W, FRAME_H
from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam

# Create a temp path for images with telemetry.
tmp_dir = '/users/anoopa/mydonkey/tmp'

# Build the outdir path.
in_path = '/users/anoopa/mydonkey/sessions/Self-7-22-1'

model = keras.models.load_model("model.hdf5")

file_count = len([f for f in os.listdir(in_path) if 'lidar' not in f])

i = 0
for filename in os.listdir(in_path):
  if 'frame' not in filename:
    continue
  image = utils.load_img(os.path.join(in_path, filename))
  # Convert to BGR, create input with batch_size: 1.
  #bgr_img = utils.bgr2rgb(image)
  #img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
  
  heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0, 
                          seed_input=image, grad_modifier=None)

  image = overlay(image, heatmap, alpha=0.7)
  im = Image.fromarray(image)
  im.save(os.path.join(tmp_dir, filename))
  i += 1
  if i % 100 == 0 or i == file_count:
    sys.stdout.write('\rwriting attention map.. %0.1f%%' % (100. * i / file_count))
    sys.stdout.flush()
  