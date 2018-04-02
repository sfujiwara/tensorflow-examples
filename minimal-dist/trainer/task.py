import json
import os
import time
import tensorflow as tf


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Get ML Engine config from environment variable
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    server = tf.train.Server(
        tf_config["cluster"],
        job_name=tf_config["task"]["type"],
        task_index=tf_config["task"]["index"]
    )
    tf.logging.debug(tf_config)
    # Parameter server
    if tf_config["task"]["type"] == "ps":
        server.join()

    # Create device function
    device_fn = tf.train.replica_device_setter(
        cluster=tf_config["cluster"],
        worker_device="/job:{0}/task:{1}".format(tf_config["task"]["type"], tf_config["task"]["index"]),
    )

    # Build graph
    with tf.Graph().as_default() as g:
        with tf.device(device_fn):
            count = tf.Variable(0, name="count")
            increment_op = tf.assign_add(count, 1)
            init_op = tf.global_variables_initializer()
            if tf_config["task"]["type"] == "master":
                tf.summary.FileWriter(logdir="summary", graph=g)
    # Start session
    with tf.Session(
        target=server.target,
        config=tf.ConfigProto(log_device_placement=False),
        graph=g,
    ) as sess:
        sess.run(init_op)
        for i in range(10):
            sess.run(increment_op)
            time.sleep(2)
            tf.logging.info(
                "{0} {1} increments count: {2}".format(
                    tf_config["task"]["type"],
                    tf_config["task"]["index"],
                    sess.run(count)
                )
            )


if __name__ == "__main__":
    main()
