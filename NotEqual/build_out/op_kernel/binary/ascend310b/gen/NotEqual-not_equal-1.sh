#!/bin/bash
echo "[Ascend310B1] Generating NotEqual_fd4d460d6f2b7994f826c07d743c6c1e ..."
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
res=$(opc $1 --main_func=not_equal --input_param=/root/ascend_op/op/NotEqual/build_out/op_kernel/binary/ascend310b/gen/NotEqual_fd4d460d6f2b7994f826c07d743c6c1e_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/NotEqual_fd4d460d6f2b7994f826c07d743c6c1e.json ; then
  echo "$2/NotEqual_fd4d460d6f2b7994f826c07d743c6c1e.json not generated!"
  exit 1
fi

if ! test -f $2/NotEqual_fd4d460d6f2b7994f826c07d743c6c1e.o ; then
  echo "$2/NotEqual_fd4d460d6f2b7994f826c07d743c6c1e.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NotEqual_fd4d460d6f2b7994f826c07d743c6c1e Done"
