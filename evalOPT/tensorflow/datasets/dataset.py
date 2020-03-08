# -*- coding: utf-8 -*-
"""Base class for evalOPT datasets."""

import tensorflow as tf


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class DataSet(object):
    """Base class for evalOPT data sets.
  Args:
    batch_size (int): The mini-batch size to use.
  Attributes:
    batch: A tuple of tensors, yielding batches of data from the dataset.
        Executing these tensors raises a ``tf.errors.OutOfRangeError`` after one
        epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
  """

    def __init__(self, data_dir, batch_size):
        """Creates a new DataSet instance.
    Args:
      batch_size (int): The mini-batch size to use.
    """
        self._batch_size = batch_size
        self._data_dir = data_dir
        self._train_dataset = self._make_train_dataset(data_dir)
        self._train_eval_dataset = self._make_train_eval_dataset(data_dir)
        self._test_dataset = self._make_test_dataset(data_dir)

        # Reinitializable iterator given types and shapes of the outputs
        # (needs to be the same for train and test of course).
        self._iterator = tf.data.Iterator.from_structure(
            self._train_dataset.output_types,
            self._train_dataset.output_shapes)
        self.batch = self._iterator.get_next()

        # Operations to switch phases (reinitialize iterator and assign value to
        # phase variable)
        self.phase = tf.Variable("train", name="phase", trainable=False)
        self.train_init_op = tf.group([
            self._iterator.make_initializer(self._train_dataset),
            tf.assign(self.phase, "train")
        ], name="train_init_op")
        self.train_eval_init_op = tf.group([
            self._iterator.make_initializer(self._train_eval_dataset),
            tf.assign(self.phase, "train_eval")
        ], name="train_eval_init_op")
        self.test_init_op = tf.group([
            self._iterator.make_initializer(self._test_dataset),
            tf.assign(self.phase, "test")
        ], name="test_init_op")

    def _make_train_dataset(self):
        """Creates the training dataset.
    Returns:
      A tf.data.Dataset instance with batches of training data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_train_eval_dataset(self):
        """Creates the train eval dataset.
    Returns:
      A tf.data.Dataset instance with batches of training eval data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_test_dataset(self):
        """Creates the test dataset.
    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")