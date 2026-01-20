current_dir=`dirname "$0"`
current_dir=`cd "$current_dir"; pwd`
root_dir=${current_dir}/../../../../../
workload_config=${root_dir}/conf/workloads/ml/libsvm.conf
. "${root_dir}/bin/functions/load_bench_config.sh"

enter_bench LibSVMDataPrepare ${workload_config} ${current_dir}
show_bannar start

rmr_hdfs $INPUT_HDFS || true
START_TIME=`timestamp`

run_spark_job com.intel.hibench.sparkbench.ml.LibSVMDataGenerator $INPUT_HDFS $NUM_EXAMPLES_SVM $NUM_FEATURES_SVM

END_TIME=`timestamp`

show_bannar finish
leave_bench

