"""Example run script using StandardRunner."""
import sys
import tensorflow as tf
import evalOPT.tensorflow as tfobs

optimizer_class = tf.train.MomentumOptimizer
hyperparams = [{
    "name": "momentum",
    "type": float,
    "default": 0.0
}, {
    "name": "use_nesterov",
    "type": bool,
    "default": False
}]
runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)

# The run method accepts all the relevant inputs.
# optimizer related hyper-parameters are preferred to be in hyperparams.
# Other options will be arguments below, e.g., data_dir, out_dir, etc.
runner.run(parent_dir=str(sys.argv[1]), testproblem="mnist_mlp", learning_rate=0.01)