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

import json
import logging
import numpy as np
import os
import socket
import subprocess
import tensorflow as tf
import time
import multiprocessing

from model import get_model

KUBERNETES = True  # Set true if on k8s. False if local testing.

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.008, "Initial learning rate.")
tf.app.flags.DEFINE_integer("steps_to_validate", 10,
                            "Validate and print loss after this many steps")
tf.app.flags.DEFINE_integer("is_sync", 1, "Synchronous updates?")
tf.app.flags.DEFINE_string("train_dir", "/output", "directory to write "
                                                   "checkpoint files")
tf.app.flags.DEFINE_integer("num_epochs", 5, "number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1024, "batch size")
tf.app.flags.DEFINE_string("CHECKPOINTS_DIRECTORY", "checkpoints",
                           "Directory to store checkpoints")
tf.app.flags.DEFINE_integer("interop", 1, "Number of interop threads")
tf.app.flags.DEFINE_integer("intraop", multiprocessing.cpu_count() - 1,
                            "Number of intraop threads")

if not KUBERNETES:
    tf.app.flags.DEFINE_string("job_name", "ps", "job name: worker or ps")
    tf.app.flags.DEFINE_integer("task_index", 0, "task index (0)")

# You can turn on the gRPC messages by setting the environment variables below
# os.environ["GRPC_VERBOSITY"]="DEBUG"
# os.environ["GRPC_TRACE"] = "all"
num_inter_op_threads = FLAGS.interop
num_intra_op_threads = FLAGS.intraop

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(num_intra_op_threads)


def main(_):

    start_time = time.time()

    MAX_STEPS = 10000  # Maximum steps to train

    logging.info("TensorFlow version: %s", tf.__version__)
    logging.info("TensorFlow git version: %s", tf.__git_version__)

    if KUBERNETES:
        tf_config_json = os.environ.get("TF_CONFIG", "{}")
        tf_config = json.loads(tf_config_json)
        logging.info("tf_config: {}".format(tf_config))

        task = tf_config.get("task", {})
        task_index = task["index"]
        job_name = task["type"]
        logging.info("task: {}".format(task))

        cluster_spec = tf_config.get("cluster", {})
        logging.info("cluster_spec: {}".format(cluster_spec))

    else:   # Local testing
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        cluster_spec = {"ps": ["localhost:2222"],
                        "worker": ["localhost:2223",
                                   "localhost:2224"]}

    worker_list = cluster_spec.get("worker", "{}")
    ps_list = cluster_spec.get("ps", "{}")

    logging.info("job_name: {}".format(job_name))
    logging.info("task_index: {}".format(task_index))

    config = tf.ConfigProto(
        inter_op_parallelism_threads=num_inter_op_threads,
        intra_op_parallelism_threads=num_intra_op_threads)

    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    is_sync = (FLAGS.is_sync == 1)  # Synchronous or asynchronous updates
    is_chief = (task_index == 0)  # Am I the chief node (always task 0)

    if job_name == "ps":

        logging.info("I am parameter server #{}. "
                     "I will join the server and will "
                     "need to be explicitly terminated when all jobs end. "
                     "Kubernetes should do this automatically."
                     "Otherwise, use CTRL-\\ for manual termination".
                     format(task_index))
        server.join()

    elif job_name == "worker":

        if is_chief:
            logging.info("I am the chief worker {} with task #{}".format(
                worker_list[task_index], task_index))
        else:
            logging.info("I am worker {} with task #{}".format(
                worker_list[task_index], task_index))

        # Graph
        worker_device = "/job:{}/task:{}".format(job_name, task_index)
        setter = tf.train. \
            replica_device_setter(ps_tasks=len(ps_list),
                                  worker_device=worker_device)
        with tf.device(setter):

            """
            BEGIN: MODEL DEFINE
            """
            input_tensor = tf.placeholder(tf.float32)
            label_tensor = tf.placeholder(tf.float32)
            model = get_model(input_tensor,
                              label_tensor,
                              FLAGS, is_chief,
                              MAX_STEPS,
                              len(worker_list))
            """
            END: MODEL DEFINE
            """

        # Monitored Training Session
        checkpoint_dir = None
        # if is_chief:
        #     checkpoint_dir = FLAGS.CHECKPOINTS_DIRECTORY
        # else:
        #     checkpoint_dir = None

        params = dict(master=server.target,
                      is_chief=is_chief,
                      config=config,
                      hooks=model["hooks"],
                      checkpoint_dir=checkpoint_dir,
                      stop_grace_period_secs=10)
        sess = tf.train.MonitoredTrainingSession(**params)

        logging.info("Starting training on worker {}".format(task_index))

        if is_chief:
            time.sleep(5)

        """
        Just predict a simple line.
        """
        slope = 8.16
        intercept = -19.71

        while not sess.should_stop():

            train_x = np.random.randn(1)*10
            train_y = slope * train_x + \
                intercept + np.random.randn(1) * 0.33

            _, result, loss, m, b, step = sess.run([model["optimizer"],
                                                    model["prediction"],
                                                    model["loss"],
                                                    model["m"], model["b"],
                                                    model["global_step"]],
                                                   feed_dict={
                input_tensor: train_x,
                label_tensor: train_y})

            logging.info("worker {}, step {}: loss = {:.4}, "
                         "target: [{}, {}], prediction: [{:.4}, {:.4}]".
                         format(task_index, step, loss,
                                slope, intercept, m, b))

        logging.info("Finished on task {}".format(task_index))

        logging.info(
            "Session from worker {} closed cleanly".format(task_index))


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    tf.app.run()
