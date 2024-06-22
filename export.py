from ultralytics import YOLO

model = YOLO('Modelos/Dolares4.pt')

model.export(format = 'onnx')