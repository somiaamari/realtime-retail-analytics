# ONNX export parity

- Weights: `models\retail_people_clean.pt`
- ONNX: `models\exports\retail_people_clean_int8.onnx`
- Precision: int8
- Validation images: 10
- Maximum absolute raw-output difference: 381.78842163
- Mean absolute raw-output difference: 5.46626005
- Detection parity (IoU >= 0.95 and score difference < 0.01): 3.3333% (1 matched; 30 PyTorch / 28 ONNX detections)
