# -*- coding: utf-8 -*-

import apache_beam as beam
import tensorflow as tf


def _create_tfrecord(element):
    feature = {
        # 'x': tf.train.Feature(float_list=tf.train.FloatList(value=element['x'])),
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=element['x'])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[element['y'].encode('utf-8')])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def main():

    input_data = [
        {'x': [ord(i) for i in 'fool'], 'y': 'fool'},
        {'x': [ord(i) for i in 'cool'], 'y': 'cool'},
        {'x': [ord(i) for i in 'hello'], 'y': 'hello'}
    ]

    print(input_data)

    options = beam.options.pipeline_options.PipelineOptions()
    options.view_as(beam.options.pipeline_options.StandardOptions).runner = "DirectRunner"

    p = beam.Pipeline(options=options)

    write_tfrecord = beam.io.WriteToTFRecord(
        'sample',
        file_name_suffix='.tfrecord'
    )

    (p | 'Read Inputs' >> beam.Create(input_data)
       | 'Create TFRecord' >> beam.Map(_create_tfrecord)
       | 'Write TFRecord' >> beam.io.Write(write_tfrecord))

    p.run().wait_until_finish()


if __name__ == '__main__':
    main()
