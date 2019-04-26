#!/bin/sh

time $SPARK_HOME/bin/spark-submit --master master_address  part3_t3_large.py partition_number output_path input_files_path
