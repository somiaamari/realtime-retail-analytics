# ONNX export parity

- Weights: `models\retail_people_clean.pt`
- ONNX: `models\exports\retail_people_clean_fp16.onnx`
- Precision: fp16
- Validation images: 10
- Maximum absolute raw-output difference: 5.38098145
- Mean absolute raw-output difference: 0.08307636
- Detection parity (IoU >= 0.95 and score difference < 0.01): 96.6667% (29 matched; 30 PyTorch / 29 ONNX detections)
