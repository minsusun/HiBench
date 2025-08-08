#!/bin/bash

current_dir=`dirname "$0"`
current_dir=`cd "$current_dir"; pwd`
root_dir=${current_dir}/../../../../../
workload_config=${root_dir}/conf/workloads/ml/svd.conf
. "${root_dir}/bin/functions/load_bench_config.sh"

enter_bench BlazeSVDPPDataPrepare ${workload_config} ${current_dir}
show_bannar start

rmr_hdfs $INPUT_HDFS || true
START_TIME=`timestamp`

run_spark_job com.intel.hibench.sparkbench.ml.BlazeSVDPPDataGenerator $INPUT_HDFS $NUM_EXAMPLES_SVD $NUM_FEATURES_SVD

END_TIME=`timestamp`

show_bannar finish
leave_bench

