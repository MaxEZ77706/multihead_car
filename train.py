from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler  # ✅ Правильно

from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm
import os

# === Классы ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === Dataset с поддержкой пробелов ===
class MultiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(folder_path):
                continue

            parts = class_folder.strip().split()
            if len(parts) != 2:
                print(f"⚠️ Неверный формат папки: {class_folder}. Пропущено.")
                continue

            color, brand = parts
            if color not in color_classes or brand not in brand_classes:
                print(f"⚠️ Неизвестный цвет или бренд: {class_folder}. Пропущено.")
                continue

            color_idx = color_classes.index(color)
            brand_idx = brand_classes.index(brand)

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, brand_idx, color_idx))

        if not self.samples:
            raise ValueError(f"❌ Нет допустимых данных в {root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, brand_label, color_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(brand_label), torch.tensor(color_label)

# === Модель ===
class MobileNetV3_MultiHead(nn.Module):
    def __init__(self, num_classes1=5, num_classes2=5):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.flatten(self.pool(self.features(dummy)))
            in_features = out.shape[1]

        self.head1 = nn.Linear(in_features, num_classes1)
        self.head2 = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head1(x), self.head2(x)

# === Главная функция ===
def main():
    dataset_root = "MultiLabel"
    batch_size = 64
    num_epochs = 3
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=os.path.join(dataset_root, "runs"))
    scaler = GradScaler(enabled=torch.cuda.is_available())

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        normalize
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = MultiHeadDataset(os.path.join(dataset_root, "train"), transform=transform_train)
    val_dataset = MultiHeadDataset(os.path.join(dataset_root, "val"), transform=transform_val_test)
    test_dataset = MultiHeadDataset(os.path.join(dataset_root, "test"), transform=transform_val_test)

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model = MobileNetV3_MultiHead().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct1 = 0
        correct2 = 0

        for inputs, labels1, labels2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
            optimizer.zero_grad()
            with autocast():
                out1, out2 = model(inputs)
                loss1 = criterion(out1, labels1)
                loss2 = criterion(out2, labels2)
                loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct1 += (out1.argmax(1) == labels1).sum().item()
            correct2 += (out2.argmax(1) == labels2).sum().item()

        train_acc1 = correct1 / len(train_loader.dataset)
        train_acc2 = correct2 / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)

        # === Валидация ===
        model.eval()
        val_correct1 = 0
        val_correct2 = 0
        with torch.no_grad():
            for inputs, labels1, labels2 in val_loader:
                inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
                with autocast():
                    out1, out2 = model(inputs)
                val_correct1 += (out1.argmax(1) == labels1).sum().item()
                val_correct2 += (out2.argmax(1) == labels2).sum().item()

        val_acc1 = val_correct1 / len(val_loader.dataset)
        val_acc2 = val_correct2 / len(val_loader.dataset)
        scheduler.step()

        # === TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/Train_Brand", train_acc1, epoch)
        writer.add_scalar("Accuracy/Train_Color", train_acc2, epoch)
        writer.add_scalar("Accuracy/Val_Brand", val_acc1, epoch)
        writer.add_scalar("Accuracy/Val_Color", val_acc2, epoch)

        avg_val_acc = (val_acc1 + val_acc2) / 2
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(dataset_root, "multihead_best.pth"))
            print(f"✅ Best model saved at epoch {epoch+1} with val_acc = {avg_val_acc:.4f}")

        print(f"📊 Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Accs: {train_acc1:.4f}, {train_acc2:.4f} | Val Accs: {val_acc1:.4f}, {val_acc2:.4f}")

    # === Тестирование ===
    model.eval()
    test_correct1 = 0
    test_correct2 = 0
    with torch.no_grad():
        for inputs, labels1, labels2 in test_loader:
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
            with autocast():
                out1, out2 = model(inputs)
            test_correct1 += (out1.argmax(1) == labels1).sum().item()
            test_correct2 += (out2.argmax(1) == labels2).sum().item()

    test_acc1 = test_correct1 / len(test_loader.dataset)
    test_acc2 = test_correct2 / len(test_loader.dataset)
    print(f"\n🎯 Test Accuracy - Brand: {test_acc1:.4f} | Color: {test_acc2:.4f}")

    torch.save(model.state_dict(), os.path.join(dataset_root, "multihead_final.pth"))
    print("✅ Final model saved as multihead_final.pth")
    writer.close()

# === Запуск безопасно с multiprocessing
if __name__ == "__main__":
    main()
