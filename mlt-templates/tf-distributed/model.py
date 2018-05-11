import tensorflow as tf

"""
Right now just returning the model and ops as a dictionary.
Next step is to move to Krishna's python class
"""


def get_model(input_tensor, label_tensor,
              FLAGS, is_chief, MAX_STEPS, num_replicas):
    """
    Simple model of a line y = mx + b
    """
    m = tf.get_variable("slope", [],
                        tf.float32,
                        initializer=tf.random_normal_initializer())

    b = tf.get_variable("intercept", [],
                        tf.float32,
                        initializer=tf.random_normal_initializer())

    # Our target model is just a line
    prediction = m * input_tensor + b

    # Calculate loss
    loss = tf.losses.mean_squared_error(label_tensor, prediction)

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

    hooks = [tf.train.StopAtStepHook(last_step=MAX_STEPS)]
    if FLAGS.is_sync:
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        hooks.append(sync_replicas_hook)

    model = {}
    model["prediction"] = prediction
    model["optimizer"] = opt
    model["hooks"] = hooks
    model["global_step"] = global_step
    model["loss"] = loss
    model["b"] = b
    model["m"] = m

    return model
