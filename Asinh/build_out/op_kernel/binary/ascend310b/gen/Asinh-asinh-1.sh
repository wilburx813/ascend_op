#!/bin/bash
echo "[Ascend310B1] Generating Asinh_5e4658cc2e6dc559f1035b8ce9270971 ..."
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
res=$(opc $1 --main_func=asinh --input_param=/root/ascend_op/op/Asinh/build_out/op_kernel/binary/ascend310b/gen/Asinh_5e4658cc2e6dc559f1035b8ce9270971_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Asinh_5e4658cc2e6dc559f1035b8ce9270971.json ; then
  echo "$2/Asinh_5e4658cc2e6dc559f1035b8ce9270971.json not generated!"
  exit 1
fi

if ! test -f $2/Asinh_5e4658cc2e6dc559f1035b8ce9270971.o ; then
  echo "$2/Asinh_5e4658cc2e6dc559f1035b8ce9270971.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Asinh_5e4658cc2e6dc559f1035b8ce9270971 Done"
