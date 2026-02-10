"""
Основной скрипт выполнения эксперимента
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys


def run_experiment(job_config, output_dir):
    """
    Запуск эксперимента по конфигурации
    """
    print(f"🧪 Запуск эксперимента: {job_config.get('name', 'unnamed')}")

    # Настройка параметров
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Параметры из задания
    epochs = job_config.get('epochs', 10)
    batch_size = job_config.get('batch_size', 32)
    latent_dim = job_config.get('latent_dim', 20)
    models_to_run = job_config.get('models', ['vae', 'iwae', 'focus_elbo'])

    print(f"📊 Параметры: epochs={epochs}, batch={batch_size}, device={device}")

    # Загрузка данных
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Здесь ваш основной код VAE
    # ... (вставьте ваш код обучения моделей)

    # Сохраняем результаты
    results = {
        "job_id": job_config.get("id", "unknown"),
        "completed_at": datetime.now().isoformat(),
        "device": str(device),
        "final_losses": {},  # Здесь будут результаты
        "training_time": 0,
        "metrics": {}
    }

    # Сохраняем в файл
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Результаты сохранены: {results_file}")
    return results