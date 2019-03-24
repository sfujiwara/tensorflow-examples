import json
import os


def delete_tf_config():

    if 'TF_CONFIG' in os.environ.keys():
        del os.environ['TF_CONFIG']


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

    if 'ps' not in tf_config['cluster'].keys():
        return

    tf_config['cluster']['evaluator'] = tf_config['cluster']['ps']
    del tf_config['cluster']['ps']

    if tf_config['task']['type'] == 'ps':
        tf_config['task']['type'] = 'evaluator'

    os.environ['TF_CONFIG'] = json.dumps(tf_config)
