# ONNX export parity

- Weights: `models\retail_people_clean.pt`
- ONNX: `models\exports\retail_people_clean.onnx`
- Precision: fp32
- Validation images: 10
- Maximum absolute raw-output difference: 0.00231934
- Mean absolute raw-output difference: 0.00002759
- Detection parity (IoU >= 0.95 and score difference < 0.01): 100.0000% (30 matched; 30 PyTorch / 30 ONNX detections)
