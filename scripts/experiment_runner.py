"""
Создание заданий для экспериментов
"""

import json
from datetime import datetime
from pathlib import Path

def create_experiment_job(name, job_type="quick", parameters=None):
    """
    Создает новое задание для эксперимента

    Args:
        name: Название задания
        job_type: Тип задания (quick, full, custom)
        parameters: Дополнительные параметры

    Returns:
        str: ID созданного задания
    """
    # Базовые конфигурации
    configs = {
        "quick": {
            "epochs": 5,
            "batch_size": 32,
            "models": ["vae", "focus_elbo"],
            "description": "Быстрый тест"
        },
        "full": {
            "epochs": 30,
            "batch_size": 128,
            "models": ["vae", "iwae", "vamp", "focus_elbo"],
            "description": "Полное сравнение"
        },
        "beta_study": {
            "epochs": 15,
            "batch_size": 64,
            "models": ["focus_elbo"],
            "beta_values": [0.001, 0.01, 0.05, 0.1, 0.2],
            "description": "Исследование гиперпараметра beta"
        }
    }

    # Получаем базовую конфигурацию
    config = configs.get(job_type, configs["quick"]).copy()

    # Добавляем пользовательские параметры
    if parameters:
        config.update(parameters)

    # Создаем ID
    job_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Полное задание
    job = {
        "id": job_id,
        "name": name,
        "type": job_type,
        "config": config,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_by": "colab_script"
    }

    # Сохраняем
    repo_dir = Path("/content/focus-vae-experiment")
    pending_dir = repo_dir / "experiments" / "jobs" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    job_file = pending_dir / f"{job_id}.json"

    with open(job_file, 'w') as f:
        json.dump(job, f, indent=2)

    print(f"✅ Задание создано: {job_id}")
    print(f"📁 Файл: {job_file}")
    print(f"📊 Конфигурация: {json.dumps(config, indent=2)}")

    return job_id

# Примеры использования
def create_demo_jobs():
    """Создание демо-заданий"""
    print("🎯 Создание демо-заданий...")

    jobs = [
        create_experiment_job("quick_test", "quick"),
        create_experiment_job("full_comparison", "full"),
        create_experiment_job("beta_research", "beta_study")
    ]

    print(f"\n✅ Создано {len(jobs)} демо-заданий")
    return jobs

if __name__ == "__main__":
    create_demo_jobs()