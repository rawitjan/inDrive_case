import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import json

# --- Конфигурация ---
INTEGRITY_MODEL_PATH = "models/integrity_model.pth"
CLEANLINESS_MODEL_PATH = "models/cleanliness_model.pth"

# Важно: классы должны быть в том же порядке, что и при обучении (алфавитном)
INTEGRITY_CLASSES = ['damaged', 'intact']  # Битый, Целый
CLEANLINESS_CLASSES = ['clean', 'dirty']  # Чистый, Грязный

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Загрузка моделей ---

def load_model(model_path, num_classes):
    """Загружает модель ResNet-50 с кастомным последним слоем."""
    model = models.resnet50()  # Не используем pretrained weights, так как загружаем свои
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Загружаем обе модели
integrity_model = load_model(INTEGRITY_MODEL_PATH, len(INTEGRITY_CLASSES))
cleanliness_model = load_model(CLEANLINESS_MODEL_PATH, len(CLEANLINESS_CLASSES))
print("Модели успешно загружены.")

# --- Функция предсказания ---

# Трансформации для входного изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image):
    """
    Принимает изображение PIL, возвращает предсказания от обеих моделей.
    """
    if image is None:
        return "Пожалуйста, загрузите изображение"

    # Предобработка изображения
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)

    # Получение предсказаний
    with torch.no_grad():
        # Модель целостности
        out_integrity = integrity_model(batch_t)
        probs_integrity = torch.nn.functional.softmax(out_integrity, dim=1)[0]

        # Модель чистоты
        out_cleanliness = cleanliness_model(batch_t)
        probs_cleanliness = torch.nn.functional.softmax(out_cleanliness, dim=1)[0]

    # Форматирование результата
    integrity_result = {
        "Статус": "Целый" if INTEGRITY_CLASSES[torch.argmax(probs_integrity)] == 'intact' else "Битый",
        "Уверенность": torch.max(probs_integrity).item()
    }

    cleanliness_result = {
        "Статус": "Чистый" if CLEANLINESS_CLASSES[torch.argmax(probs_cleanliness)] == 'clean' else "Грязный",
        "Уверенность": torch.max(probs_cleanliness).item()
    }

    # Возвращаем красивый JSON
    final_result = {
        "Целостность": integrity_result,
        "Чистота": cleanliness_result
    }

    return final_result


# --- Запуск интерфейса Gradio ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите фото автомобиля"),
    outputs=gr.JSON(label="Результат анализа"),
    title="Анализ состояния автомобиля",
    description="Загрузите фотографию автомобиля, чтобы определить его целостность (битый/целый) и чистоту (грязный/чистый).",
    examples=[["data/integrity/train/damaged/0c265959a64be5b9dc10106830be0351_jpeg.rf.8c863da57651bd2584fc83d7af583459.jpg"],
              ["data/integrity/train/intact/00001.jpg"],
              ["data/cleanliness/train/clean/download.jpg"],
              ["data/cleanliness/train/dirty/download (2)(1).jpg"]]  # Добавьте сюда пути к примерам
)

if __name__ == "__main__":
    iface.launch()
