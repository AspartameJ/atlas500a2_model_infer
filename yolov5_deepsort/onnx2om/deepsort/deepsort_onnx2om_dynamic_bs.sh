string="1"
for i in {2..99};do string="$string,$i";done
string="$string,100"
echo $string
atc --model=ckpt1.onnx --framework=5 --output=ckpt1_bs1-100 --input_format=NCHW --input_shape="conv.0:-1,3,128,64" --dynamic_batch_size=$string -log=info --soc_version=Ascend310B1
