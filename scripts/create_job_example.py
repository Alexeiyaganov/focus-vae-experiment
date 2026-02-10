"""
Создание задания для эксперимента прямо из Colab
"""

import json
from datetime import datetime
from github_connector import GitHubConnector


def create_job_from_colab():
    """Создание задания прямо из Colab"""

    print("🎯 Создание нового задания для эксперимента")
    print("=" * 50)

    # Настраиваем коннектор
    connector = GitHubConnector()
    connector.clone_or_pull_repository()

    # Данные задания
    job_id = f"colab_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    job_config = {
        "id": job_id,
        "name": "Full Comparison from Colab",
        "description": "Полное сравнение всех методов, запущено из Colab",
        "created_from": "colab",
        "created_at": datetime.now().isoformat(),
        "status": "pending",

        # Параметры эксперимента
        "parameters": {
            "epochs": 30,
            "batch_size": 128,
            "latent_dim": 32,
            "learning_rate": 3e-4,

            # Модели для сравнения
            "models": [
                {
                    "name": "Standard VAE",
                    "type": "vae"
                },
                {
                    "name": "IWAE (K=5)",
                    "type": "iwae",
                    "k_samples": 5
                },
                {
                    "name": "Focus-ELBO (наш)",
                    "type": "focus_elbo",
                    "k_samples": 5,
                    "focus_steps": 2,
                    "beta": 0.01
                }
            ],

            # Датсет
            "dataset": "MNIST",
            "split": {
                "train": 50000,
                "val": 10000,
                "test": 10000
            },

            # Визуализация
            "visualizations": [
                "convergence_plot",
                "generation_samples",
                "latent_space",
                "reconstruction_comparison"
            ]
        }
    }

    # Сохраняем задание
    jobs_dir = connector.repo_dir / "experiments" / "jobs" / "pending"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_file = jobs_dir / f"{job_id}.json"

    with open(job_file, 'w') as f:
        json.dump(job_config, f, indent=2)

    print(f"✅ Задание создано: {job_id}")
    print(f"📁 Файл: {job_file}")

    # Коммитим и пушим
    print("\n💾 Сохраняем задание в GitHub...")
    connector.push_results(f"Colab: Created new job {job_id}")

    print("\n🎯 Задание готово к выполнению!")
    print("Colab Worker автоматически обнаружит и выполнит его.")

    return job_id


# Запуск
if __name__ == "__main__":
    create_job_from_colab()