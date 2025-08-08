import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

# === 1. Загрузите правильную архитектуру ===
model = mobilenet_v3_large(pretrained=False)
model.classifier[3] = nn.Linear(1280, 5)  # замените 5 на количество ваших классов

# === 2. Загрузите веса ===
model.load_state_dict(torch.load("color_detection_best_mobilenetv3.pth", map_location="cpu"))
model.eval()

# === 3. Создайте dummy input ===
dummy_input = torch.randn(1, 3, 224, 224)

# === 4. Экспорт в ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    "color_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("✅ Успешно конвертировано в color_model.onnx")
