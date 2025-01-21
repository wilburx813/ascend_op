#!/bin/bash
echo "[Ascend310B1] Generating LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=log_sum_exp --input_param=/root/ascend_op/op/LogSumExp/build_out/op_kernel/binary/ascend310b/gen/LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c.json ; then
  echo "$2/LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c.json not generated!"
  exit 1
fi

if ! test -f $2/LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c.o ; then
  echo "$2/LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating LogSumExp_3c7d8156ec319d7a7cb7332ecef5409c Done"
