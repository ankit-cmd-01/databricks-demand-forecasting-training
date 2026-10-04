"""Spark session helper.

Notebooks provide a global `spark`; plain .py files run as jobs do not.
Classes receive the session through __init__ instead of using a global.
"""
from pyspark.sql import SparkSession


def get_spark():
    return SparkSession.builder.getOrCreate()