import json
import os


# tf_config = {
#     "cluster": {
#         "master": ["127.0.0.1:2222"]
#     },
#     "environment": "cloud",
#     "task": {
#         "index": 0,
#         "cloud": "d9591dfbdf136e350-ml",
#         "type": "master"
#     }
# }


def replace_master_with_chief():

    tf_config = os.environ.get('TF_CONFIG')

    if tf_config is None:
        return

    tf_config = json.loads(tf_config)
    tf_config['cluster']['chief'] = tf_config['cluster']['master']
    del tf_config['cluster']['master']

    if tf_config['task']['type'] == 'master':
        tf_config['task']['type'] = 'chief'

    os.environ['TF_CONFIG'] = json.dumps(tf_config)


def replace_ps_with_evaluator():

    tf_config = os.environ.get('TF_CONFIG')

    if tf_config is None:
        return

    tf_config = json.loads(tf_config)
    tf_config['cluster']['evaluator'] = tf_config['cluster']['ps']
    del tf_config['cluster']['ps']

    if tf_config['task']['type'] == 'ps':
        tf_config['task']['type'] = 'evaluator'

    os.environ['TF_CONFIG'] = json.dumps(tf_config)
