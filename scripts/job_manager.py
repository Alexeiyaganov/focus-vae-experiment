"""
Управление очередью экспериментов
"""

import json
import os
from datetime import datetime
from pathlib import Path


class JobManager:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
        self.jobs_dir = self.repo_path / "experiments" / "jobs"

        # Создаем структуру папок
        for status in ["pending", "running", "completed", "failed"]:
            (self.jobs_dir / status).mkdir(parents=True, exist_ok=True)

    def create_job(self, name, config):
        """Создание нового задания"""
        job_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job = {
            "id": job_id,
            "name": name,
            "config": config,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "assigned_to": None,
            "completed_at": None,
            "results": None
        }

        # Сохраняем задание
        job_file = self.jobs_dir / "pending" / f"{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job, f, indent=2)

        print(f"✅ Задание создано: {job_id}")
        return job_id

    def get_next_job(self):
        """Получение следующего задания"""
        pending_dir = self.jobs_dir / "pending"

        for job_file in pending_dir.glob("*.json"):
            with open(job_file, 'r') as f:
                job = json.load(f)

            # Перемещаем в running
            new_path = self.jobs_dir / "running" / job_file.name
            job_file.rename(new_path)

            job["status"] = "running"
            job["assigned_to"] = "colab_worker"

            # Обновляем файл
            with open(new_path, 'w') as f:
                json.dump(job, f, indent=2)

            return job

        return None

    def update_job_status(self, job_id, status, results=None):
        """Обновление статуса задания"""
        running_file = self.jobs_dir / "running" / f"{job_id}.json"

        if not running_file.exists():
            print(f"❌ Задание {job_id} не найдено в running")
            return False

        with open(running_file, 'r') as f:
            job = json.load(f)

        job["status"] = status
        job["completed_at"] = datetime.now().isoformat() if status in ["completed", "failed"] else None
        job["results"] = results

        # Перемещаем в соответствующую папку
        if status == "completed":
            new_dir = self.jobs_dir / "completed"
        elif status == "failed":
            new_dir = self.jobs_dir / "failed"
        else:
            new_dir = self.jobs_dir / "running"

        new_dir.mkdir(exist_ok=True)
        new_path = new_dir / f"{job_id}.json"

        with open(new_path, 'w') as f:
            json.dump(job, f, indent=2)

        # Удаляем старый файл
        if new_path != running_file:
            running_file.unlink(missing_ok=True)

        print(f"✅ Статус обновлен: {job_id} → {status}")
        return True


# Пример создания заданий
def create_example_jobs():
    """Создание примеров заданий"""
    manager = JobManager()

    # Быстрый тест
    manager.create_job("quick_test", {
        "epochs": 5,
        "batch_size": 32,
        "models": ["vae", "iwae", "focus_elbo"],
        "latent_dim": 20,
        "compute_mode": "fast"
    })

    # Полный эксперимент
    manager.create_job("full_comparison", {
        "epochs": 30,
        "batch_size": 128,
        "models": ["vae", "iwae", "vamp", "focus_elbo"],
        "latent_dim": 32,
        "k_samples": 5,
        "vamp_components": 500,
        "compute_mode": "gpu"
    })

    # Исследование гиперпараметров
    manager.create_job("beta_study", {
        "epochs": 15,
        "batch_size": 64,
        "models": ["focus_elbo"],
        "beta_values": [0.001, 0.01, 0.05, 0.1, 0.2],
        "focus_steps": [1, 2, 3],
        "compute_mode": "gpu"
    })


if __name__ == "__main__":
    create_example_jobs()