#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# For all outputs:
#  1) Target field
#  2) Autoencoder output
#  3) scatter plot

import tensorflow as tf

import sys
sys.path.append('/home/mo-sbrown/philip1/DCVAE_Climate_sjb1/cpm1')
from ML_models.mk2.makeDataset import getDataset
from ML_models.mk2.autoencoderModel import getModel

from ML_models.mk2.gmUtils import plotValidationField

from specify import specification

specification["strategy"] = (
    tf.distribute.get_strategy()
)  # No distribution for simple validation

# I don't need all the messages about a missing font (on Isambard)
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=5)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1990)
parser.add_argument(
    "--day", help="Test day", type=int, required=False, default=786
)
parser.add_argument(
    "--training",
    help="Use training data (not test)",
    default=False,
    action="store_true",
)
args = parser.parse_args()

purpose = "Test"
if args.training:
    purpose = "Train"
# Go through data and get the desired month
dataset = (
    getDataset(specification, purpose=purpose)
    .shuffle(specification["shuffleBufferSize"])
    .batch(1)
)
input = None
year = None
month = None
for batch in dataset:
    dateStr = tf.strings.split(batch[0][0][0], sep="/")[-1].numpy()
#    year = int(dateStr[:4])
#    month = int(dateStr[5:7])
#    if (args.month is None or month == args.month) and (
#        args.year is None or year == args.year
#    ):
#        input = batch
#        break
    year = int(dateStr[10:14]) # int(fN[:4])
    idot    = str(dateStr).find('.')
    day     = int(dateStr[15:(idot-2)]) # int(fN[5:7])
    if (day == args.day) and (year == args.year):
        print(dateStr, idot, year, day)
        input = batch
        break

if input is None:
    raise Exception("%04d-%02d not in %s dataset" % (year, day, purpose))

autoencoder = getModel(specification, args.epoch)
print("getModel success")

# Get autoencoded tensors
output = autoencoder.call(input, training=False)

# Make the plot
plotValidationField(specification, input, output, year, day, "comparison-4c-gpu.webp")
