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
from tensorflow import keras as K

"""
Right now just returning the model and ops as a dictionary.
Next step is to move to Krishna's python class
"""


def get_model(data, FLAGS, num_replicas):

    is_chief = FLAGS.task_index == 0

    with tf.variable_scope("Inf_Graph", reuse=tf.AUTO_REUSE):

        input_tensor = tf.placeholder(tf.float32,
                                      [None] + list(data["train_shape"][1:]))
        label_tensor = tf.placeholder(tf.float32,
                                      [None, data["num_classes"]])

        w_init = None  # Default is Glorot if none specified

        conv1 = tf.layers.conv2d(inputs=input_tensor,
                                 filters=16,
                                 kernel_size=5,
                                 kernel_initializer=w_init,
                                 activation=tf.nn.relu)

        pooL1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=2,
                                        strides=2)

        conv2 = tf.layers.conv2d(inputs=pooL1,
                                 filters=32,
                                 kernel_size=5,
                                 kernel_initializer=w_init,
                                 activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=2,
                                        strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=3,
                                 kernel_initializer=w_init,
                                 activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=2,
                                        strides=2)

        flatten_opt = tf.layers.Flatten()(pool2)

        fc1 = tf.layers.dense(inputs=flatten_opt,
                              units=2048,
                              kernel_initializer=w_init,
                              activation=tf.nn.relu)

        fc2 = tf.layers.dense(inputs=fc1,
                              units=1024,
                              kernel_initializer=w_init,
                              activation=tf.nn.relu)

        prediction = tf.layers.dense(inputs=fc2,
                                     units=data["num_classes"],
                                     kernel_initializer=w_init,
                                     activation=tf.nn.softmax)
    # Calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction,
        labels=label_tensor)
    loss = tf.reduce_mean(cross_entropy)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                  tf.argmax(label_tensor, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create an optimizer then wrap it with SynceReplicasOptimizer
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # global_step tells the graph where it is in training
    global_step = tf.Variable(0,
                              dtype=tf.int32,
                              trainable=False,
                              name="global_step")

    if FLAGS.is_sync:
        optimizer = tf.train.\
            SyncReplicasOptimizer(optimizer,
                                  replicas_to_aggregate=num_replicas,
                                  total_num_replicas=num_replicas)

    opt = optimizer.minimize(
        loss, global_step=global_step)  # averages gradients

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.steps_to_train)]
    if FLAGS.is_sync:
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        hooks.append(sync_replicas_hook)

    model = {}
    model["input_tensor"] = input_tensor
    model["label_tensor"] = label_tensor
    model["prediction"] = prediction
    model["optimizer"] = opt
    model["hooks"] = hooks
    model["global_step"] = global_step
    model["loss"] = loss
    model["accuracy"] = accuracy

    return model
