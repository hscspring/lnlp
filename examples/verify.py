import onnxruntime as ort
import tensorflow as tf
import numpy as np


onnx_model_file = "./onnxruntime/models/mcls/100/model.onnx"
tf_model_dir = "./tf-serving/models/mcls/100/"

input1 = np.zeros((1, 60), dtype=np.int32)
sess = ort.InferenceSession(onnx_model_file, providers=[])

results_ort = sess.run(["dense"], {"input": input1})[0]
print(results_ort)

for input_meta in sess.get_inputs():
    print(input_meta)
for output_meta in sess.get_outputs():
    print(output_meta)

model = tf.saved_model.load(tf_model_dir)
results_tf = model(input1)
print(results_tf)

for ort_res, tf_res in zip(results_ort, results_tf):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
