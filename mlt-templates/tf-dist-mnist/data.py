#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#


import tensorflow as tf
import numpy as np
import logging


def load_data():

    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist. \
        load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train = x_train / 255.0  # Scale everything between 0 and 1
    x_test = x_test / 255.0  # Scale everything between 0 and 1
    num_classes = 10  # 10 classes for MNIST (0-9)

    # One-hot encode the labels so that we can perform categorical
    # cross-entropy loss
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    data = {}
    data["x_train"] = x_train
    data["y_train"] = y_train
    data["x_test"] = x_test
    data["y_test"] = y_test

    data["train_shape"] = x_train.shape
    data["test_shape"] = x_test.shape
    data["num_train"] = x_train.shape[0]
    data["num_test"] = x_test.shape[0]
    data["num_classes"] = num_classes
    data["name"] = "MNIST"

    logging.info("Data loaded")

    return data
