import tensorflow as tf
import numpy as np

# Get flags from the command line
def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("train_size", 55000, "training size")
    flags.DEFINE_integer("batch_size", 100, "batch size")
    flags.DEFINE_integer("display_step", 1000, "display step")
    flags.DEFINE_integer("n_iterations", 1000001, "max iterations")
    return FLAGS