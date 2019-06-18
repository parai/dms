# how to generate facenet.tflite

* 0 [web facenet-on-mobile](https://medium.com/@tomdeore/facenet-on-mobile-cb6aebe38505)
* 1 [github facenet_mtcnn_to_mobile](https://github.com/parai/facenet_mtcnn_to_mobile)
* 2 [toco & python toco](https://blog.csdn.net/qq_16564093/article/details/78996563)

1 is the best way.

```sh
bazel build tensorflow/lite/toco:toco
./bazel-bin/tensorflow/lite/toco/toco \
  --input_file=facenet.pb \
  --output_file=facenet.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape="1,160,160,3" \
  --input_array=input \
  --output_array=embeddings \
  --allow_custom_ops
```
