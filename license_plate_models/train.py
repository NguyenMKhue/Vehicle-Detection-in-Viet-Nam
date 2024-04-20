from ultralytics import YOLO

model = YOLO("license_plate_detector.pt")

result = model.train(data = "config.yaml", epochs=6,device="mps")


def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

model = YOLO("license_plate_detector.pt")
model.add_callback("on_train_start", freeze_layer)
result = model.train(data = "config.yaml", epochs=6,device="mps")