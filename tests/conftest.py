import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Small local Spark session shared by all tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("unit-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()