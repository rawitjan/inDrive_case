import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import random
import matplotlib.pyplot as plt

INTEGRITY_MODEL_PATH = "models/integrity_model.pth"
CLEANLINESS_MODEL_PATH = "models/cleanliness_model.pth"
TIRES_MODEL_PATH = "models/tires_model.pth"
GLASS_MODEL_PATH = "models/glass_model.pth"
VIEWPOINT_MODEL_PATH = "models/viewpoint_model.pth"

INTEGRITY_CLASSES = ['damaged', 'intact']
CLEANLINESS_CLASSES = ['clean', 'dirty']
TIRES_CLASSES = ['flat', 'ok']
GLASS_CLASSES = ['cracked', 'ok']
VIEWPOINT_CLASSES = ['front', 'rear', 'side']

DEVICE = torch.device("cpu")


# --- 2. ЗАГРУЗКА МОДЕЛЕЙ ---
def load_model(model_path, num_classes):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"!!! ВНИМАНИЕ: Файл модели не найден: {model_path}. Будет использоваться необученная модель.")
    model.to(DEVICE)
    model.eval()
    return model


print("Загрузка моделей...")
models_to_load = {
    "integrity": (INTEGRITY_MODEL_PATH, len(INTEGRITY_CLASSES)),
    "cleanliness": (CLEANLINESS_MODEL_PATH, len(CLEANLINESS_CLASSES)),
    "tires": (TIRES_MODEL_PATH, len(TIRES_CLASSES)),
    "glass": (GLASS_MODEL_PATH, len(GLASS_CLASSES)),
    "viewpoint": (VIEWPOINT_MODEL_PATH, len(VIEWPOINT_CLASSES)),
}
ml_models = {name: load_model(path, num_cls) for name, (path, num_cls) in models_to_load.items()}


preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 3. ЛОГИКА АНАЛИЗА И ВИЗУАЛИЗАЦИИ ---
def predict_real(model, image, class_names):
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
    with torch.no_grad():
        out = model(batch_t)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
    pred_class = class_names[torch.argmax(probs)]
    confidence = torch.max(probs).item()
    return pred_class, confidence


def create_score_chart(score, status):
    if score is None: return None

    fig, ax = plt.subplots(figsize=(3, 3), facecolor='#F9FAFB')
    ax.set_aspect('equal')

    color_map = {"Отличное состояние": '#2ca02c', "Требует внимания": '#ff7f0e', "Критическое состояние": '#d62728'}
    color = color_map.get(status, '#2ca02c')

    ax.pie([score, 100 - score], startangle=90, colors=[color, '#e6e6e6'], wedgeprops=dict(width=0.4))
    ax.text(0, 0, f"{score}", ha='center', va='center', fontsize=28, fontweight='bold', color=color)

    plt.tight_layout()
    return fig


def create_status_alert(status):
    color_map = {"Отличное состояние": 'green', "Требует внимания": 'orange', "Критическое состояние": 'red'}
    icon_map = {"Отличное состояние": '✅', "Требует внимания": '⚠️', "Критическое состояние": '❗️'}

    color = color_map.get(status, 'grey')
    icon = icon_map.get(status, '❓')

    return f"""<div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white; text-align: center; font-family: sans-serif;">
               <h2 style="margin: 0;">{icon} {status}</h2></div>"""


def calculate_overall_score(analysis_results):
    score = 100
    issues, penalties = [], {'integrity': {'damaged': 40}, 'cleanliness': {'dirty': 10}, 'tires': {'flat': 20},
                             'glass': {'cracked': 35}}
    readable_names = {'integrity': 'Повреждения кузова', 'cleanliness': 'Загрязнение', 'tires': 'Проблема с шинами',
                      'glass': 'Трещины на стекле'}

    for key, result in analysis_results.items():
        if key in penalties and result['status'] in penalties[key]:
            score -= penalties[key][result['status']]
            issues.append(readable_names[key])

    score = max(0, score)
    if score >= 90:
        status = "Отличное состояние"
    elif 90 > score >= 60:
        status = "Требует внимания"
    else:
        status = "Критическое состояние"
    return score, status, issues


def generate_recommendation_with_gemma(score, issues):
    """(Симуляция) Генерация рекомендаций."""
    # ... (логика генерации остается без изменений) ...
    if not issues: return "Автомобиль в отличном состоянии. Рекомендаций нет. Счастливого пути!"
    prompt_details = f"Общая оценка: {score}/100. Обнаружены проблемы: {', '.join(issues)}."
    print(f"\n--- ПРОМПТ ДЛЯ GEMMA ---\n{prompt_details}\n-----------------------\n")
    recommendation = "Здравствуйте! Наша система провела анализ состояния вашего автомобиля.\n\n"
    if "Проблема с шинами" in issues: recommendation += "❗️ **Критическое замечание:** Обнаружена возможная проблема с шиной. **Немедленно проверьте давление в шинах перед поездкой для вашей безопасности.**\n\n"
    if "Трещины на стекле" in issues: recommendation += "⚠️ **Важное замечание:** Замечены трещины на стекле. Это может ухудшить обзор. Рекомендуем обратиться в сервис.\n\n"
    if "Повреждения кузова" in issues: recommendation += "ℹ️ **Информация:** Есть повреждения кузова. Рекомендуем запланировать ремонт.\n\n"
    if "Загрязнение" in issues: recommendation += "ℹ️ **Информация:** Автомобиль выглядит грязным. Рекомендуем посетить мойку.\n\n"
    return recommendation + "Спасибо за заботу о безопасности и качестве сервиса!"


def full_analysis(image):
    """Основная функция, запускающая весь пайплайн анализа."""
    if image is None: return None, "Загрузите изображение", None, {}, "Загрузите изображение"

    results = {
        'integrity': predict_real(ml_models['integrity'], image, INTEGRITY_CLASSES),
        'cleanliness': predict_real(ml_models['cleanliness'], image, CLEANLINESS_CLASSES),
        'tires': predict_real(ml_models['tires'], image, TIRES_CLASSES),
        'glass': predict_real(ml_models['glass'], image, GLASS_CLASSES),
        'viewpoint': predict_real(ml_models['viewpoint'], image, VIEWPOINT_CLASSES),
    }

    internal_results = {k: {'status': v[0]} for k, v in results.items()}
    score, status, issues = calculate_overall_score(internal_results)

    chart = create_score_chart(score, status)
    status_alert = create_status_alert(status)
    recommendation = generate_recommendation_with_gemma(score, issues)

    detailed_report = {
        'Целостность': {'статус': results['integrity'][0], 'уверенность': f"{results['integrity'][1]:.2f}"},
        'Чистота': {'статус': results['cleanliness'][0], 'уверенность': f"{results['cleanliness'][1]:.2f}"},
        'Шины': {'статус': results['tires'][0], 'уверенность': f"{results['tires'][1]:.2f}"},
        'Стекла': {'статус': results['glass'][0], 'уверенность': f"{results['glass'][1]:.2f}"},
        'Ракурс': {'статус': results['viewpoint'][0], 'уверенность': f"{results['viewpoint'][1]:.2f}"}
    }

    return chart, status_alert, recommendation, detailed_report, image


# --- 4. ИНТЕРФЕЙС ДАШБОРДА ---
EXAMPLES_DIR = "examples"
example_images = []
if os.path.exists(EXAMPLES_DIR):
    all_examples = [os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR) if
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    example_images = random.sample(all_examples, min(5, len(all_examples)))

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Дашборд Анализа Авто") as demo:
    gr.Markdown("# 🚘 Дашборд комплексного анализа состояния автомобиля")
    gr.Markdown(
        "Загрузите фото для получения полной оценки, детализации по параметрам и автоматических рекомендаций водителю.")

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="pil", label="Фотография автомобиля")
            submit_btn = gr.Button("Провести анализ", variant="primary", scale=1)
            gr.Examples(examples=example_images, inputs=input_image, label="Случайные примеры для анализа")

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    output_score_chart = gr.Plot(label="Общая оценка")
                with gr.Column(scale=2, min_width=300):
                    output_status = gr.Markdown(label="Статус автомобиля")

            output_recommendation = gr.Markdown(label="Рекомендации для водителя")
            output_details = gr.JSON(label="Подробный отчет моделей")

    submit_btn.click(
        fn=full_analysis,
        inputs=input_image,
        outputs=[output_score_chart, output_status, output_recommendation, output_details, input_image]
    )

if __name__ == "__main__":
    demo.launch(debug=True)

