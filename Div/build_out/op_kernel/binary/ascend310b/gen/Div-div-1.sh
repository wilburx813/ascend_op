#!/bin/bash
echo "[Ascend310B1] Generating Div_c2ca015ab87214892dd44c128a74490b ..."
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
res=$(opc $1 --main_func=div --input_param=/root/ascend_op/op/Div/build_out/op_kernel/binary/ascend310b/gen/Div_c2ca015ab87214892dd44c128a74490b_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Div_c2ca015ab87214892dd44c128a74490b.json ; then
  echo "$2/Div_c2ca015ab87214892dd44c128a74490b.json not generated!"
  exit 1
fi

if ! test -f $2/Div_c2ca015ab87214892dd44c128a74490b.o ; then
  echo "$2/Div_c2ca015ab87214892dd44c128a74490b.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Div_c2ca015ab87214892dd44c128a74490b Done"
