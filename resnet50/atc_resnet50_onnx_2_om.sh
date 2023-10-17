atc --model=resnet50.onnx --framework=5 --output=resnet50 --input_format=NCHW --input_shape="module.backbone.conv1:1,3,256,192" --log=info --soc_version=Ascend310B1
