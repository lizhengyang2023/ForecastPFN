"""
Module containing constants for synthetic data generation
"""

import tensorflow as tf
from prior_constants import BASE_START, BASE_END, CONTEXT_LENGTH

PRODUCT_SCHEMA = {
    "doc": "Timeseries sample",
    "name": "TimeseriesSample",
    "type": "record",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "ts", "type": {"type": "int", "logicalType": "date"}},
        {"name": "y", "type": ["null", "float"]},
        {"name": "noise", "type": ["float"]}
    ],
}

TF_SCHEMA = {
    "id": tf.io.FixedLenFeature([], dtype=tf.string),
    "ts": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.int64),
    "y": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32),
    "noise": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32)
}
