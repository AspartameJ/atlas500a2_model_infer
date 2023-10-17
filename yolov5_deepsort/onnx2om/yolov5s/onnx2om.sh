##### 参数设置
GETOPT_ARGS=`getopt -o 'h' -al tag:,model:,nms_mode:,quantify:,bs:,soc:,with_aipp: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        --tag) tag=$2; shift 2;;
        --model) model=$2; shift 2;;
        --nms_mode) nms_mode=$2; shift 2;;
        --quantify) quantify=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --with_aipp) with_aipp=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $tag ]]; then tag=6.1; fi
if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $nms_mode ]]; then nms_mode=nms_op; fi
if [[ -z $quantify ]]; then quantify=False; fi
if [[ -z $bs ]]; then bs=4; fi
if [[ -z $soc ]]; then soc=Ascend310P3; fi
if [[ -z $with_aipp ]]; then with_aipp=False; fi

args_info="=== onnx2om args === \n tag: $tag \n model: $model \n nms_mode: $nms_mode \n quantify: $quantify \n bs: $bs \n soc: $soc \n with_aipp: $with_aipp"
echo -e $args_info

##### 量化配置（可选）
quantify_cfg=""
if [[ $quantify == True ]] ; then
    quantify_cfg="--compression_optimize_conf=common/atc_cfg/compression_${nms_mode}.cfg"
fi

##### 方式一 nms后处理脚本
if [[ $nms_mode == nms_script ]] ; then
    if [[ $with_aipp == True ]] ; then
        echo "nms后处理脚本 + aipp前处理"
        atc --model=${model}.onnx --output=${model}_bs${bs}_aipp \
             --framework=5 --input_format=NCHW --soc_version=${soc} --log=error \
             --input_shape="images:${bs},3,640,640" \
             --enable_small_channel=1 --insert_op_conf=aipp.cfg --output_type=FP16 \
             --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance \
             --fusion_switch_file=common/atc_cfg/fusion.cfg ${quantify_cfg} || exit 1
    else
        echo "nms后处理脚本"
        atc --model=${model}.onnx --output=${model}_bs${bs} \
             --framework=5 --input_format=NCHW --soc_version=${soc} --log=error \
             --input_shape="images:${bs},3,640,640" \
             --input_fp16_nodes="images" --output_type=FP16 \
             --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance \
             --fusion_switch_file=common/atc_cfg/fusion.cfg ${quantify_cfg} || exit 1
    fi
fi

##### 方式二 nms后处理算子
if [[ $nms_mode == nms_op ]] ; then
    if [[ $with_aipp == True ]] ; then
        echo "nms后处理算子 + aipp前处理"
        atc --model=${model}.onnx --output=${model}_bs${bs}_aipp \
             --framework=5 --input_format=NCHW --soc_version=${soc} --log=error \
             --input_shape="images:${bs},3,640,640;img_info:${bs},4" \
             --enable_small_channel=1 --insert_op_conf=aipp.cfg --output_type=FP16 \
             --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance \
             --fusion_switch_file=common/atc_cfg/fusion.cfg ${quantify_cfg} || exit 1
    else
        echo "nms后处理算子"
        atc --model=${model}.onnx --output=${model}_bs${bs} \
             --framework=5 --input_format=NCHW --soc_version=${soc} --log=error \
             --input_shape="images:${bs},3,640,640;img_info:${bs},4" \
             --input_fp16_nodes="images;img_info" --output_type=FP16 \
             --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance \
             --fusion_switch_file=common/atc_cfg/fusion.cfg ${quantify_cfg} || exit 1
    fi
fi

echo -e "onnx导出om模型 Success \n"
