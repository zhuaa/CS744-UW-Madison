#!/bin/sh


time $SPARK_HOME/bin/spark-submit --master master_address part3_t2_small.py partition_number input_file_path
