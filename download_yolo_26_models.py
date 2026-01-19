from ultralytics import YOLO

for model_name in ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]:
    YOLO(f"./Models/{model_name}.pt")
    print(f"{model_name}.pt download successful")
    print("-"*50)
