# ONNX export parity

- Weights: `models\retail_people_clean.pt`
- ONNX: `models\exports\retail_people_clean_int8_signed.onnx`
- Precision: int8
- Validation images: 10
- Maximum absolute raw-output difference: 176.18914795
- Mean absolute raw-output difference: 5.83061585
- Detection parity (IoU >= 0.95 and score difference < 0.01): 0.0000% (0 matched; 30 PyTorch / 44 ONNX detections)
