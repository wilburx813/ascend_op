#!/bin/bash
echo "[Ascend310B1] Generating IsClose_4b13970aa76fcc71c1b40ffb65e26536 ..."
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
res=$(opc $1 --main_func=is_close --input_param=/root/ascend_op/op/IsClose/build_out/op_kernel/binary/ascend310b/gen/IsClose_4b13970aa76fcc71c1b40ffb65e26536_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/IsClose_4b13970aa76fcc71c1b40ffb65e26536.json ; then
  echo "$2/IsClose_4b13970aa76fcc71c1b40ffb65e26536.json not generated!"
  exit 1
fi

if ! test -f $2/IsClose_4b13970aa76fcc71c1b40ffb65e26536.o ; then
  echo "$2/IsClose_4b13970aa76fcc71c1b40ffb65e26536.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating IsClose_4b13970aa76fcc71c1b40ffb65e26536 Done"
