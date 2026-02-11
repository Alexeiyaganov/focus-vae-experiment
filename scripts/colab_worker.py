"""
Простой Colab Worker для выполнения заданий
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime


class ColabWorker:
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.repo_dir = Path("/content/focus-vae-experiment")

        print("👷 Colab Worker инициализирован")
        print(f"⏱️  Проверка заданий каждые {check_interval} секунд")

    def setup(self):
        """Настройка рабочего окружения"""
        # Создаем необходимые папки
        folders = [
            "experiments/jobs/pending",
            "experiments/jobs/running",
            "experiments/jobs/completed",
            "experiments/jobs/failed",
            "experiments/results"
        ]

        for folder in folders:
            path = self.repo_dir / folder
            path.mkdir(parents=True, exist_ok=True)

        print("📁 Структура папок создана")
        return True

    def check_jobs(self):
        """Проверяет наличие новых заданий"""
        pending_dir = self.repo_dir / "experiments" / "jobs" / "pending"

        if not pending_dir.exists():
            return None

        job_files = list(pending_dir.glob("*.json"))

        if not job_files:
            return None

        # Берем первое задание
        job_file = job_files[0]

        try:
            with open(job_file, 'r') as f:
                job = json.load(f)

            print(f"🎯 Найдено задание: {job.get('id', 'unknown')}")
            return job

        except Exception as e:
            print(f"❌ Ошибка чтения задания: {e}")
            return None

    def run_simple_experiment(self, job):
        """Простой эксперимент для демонстрации"""
        job_id = job.get('id', 'demo')

        print(f"🚀 Запуск эксперимента {job_id}...")

        # Имитация работы
        time.sleep(2)

        # Создаем результаты
        results = {
            "job_id": job_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "loss": 123.45,
                "accuracy": 0.95,
                "training_time": 120
            },
            "message": "Эксперимент выполнен успешно (демо)"
        }

        # Сохраняем результаты
        results_dir = self.repo_dir / "experiments" / "results" / job_id
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Результаты сохранены: {results_dir}")
        return True

    def process_job(self, job):
        """Обработка одного задания"""
        job_id = job.get('id', 'unknown')

        print(f"\n📌 Обработка задания: {job_id}")

        # 1. Перемещаем в running
        pending_file = self.repo_dir / "experiments" / "jobs" / "pending" / f"{job_id}.json"
        running_file = self.repo_dir / "experiments" / "jobs" / "running" / f"{job_id}.json"

        if pending_file.exists():
            running_file.parent.mkdir(parents=True, exist_ok=True)
            pending_file.rename(running_file)

        # 2. Выполняем эксперимент
        try:
            success = self.run_simple_experiment(job)

            # 3. Перемещаем в completed/failed
            if success:
                dest_dir = "completed"
                status = "completed"
            else:
                dest_dir = "failed"
                status = "failed"

            dest_file = self.repo_dir / "experiments" / "jobs" / dest_dir / f"{job_id}.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Обновляем статус
            job["status"] = status
            job["completed_at"] = datetime.now().isoformat()

            with open(dest_file, 'w') as f:
                json.dump(job, f, indent=2)

            # Удаляем из running
            if running_file.exists():
                running_file.unlink()

            print(f"✅ Задание {job_id} перемещено в {dest_dir}")

            return success

        except Exception as e:
            print(f"❌ Ошибка выполнения: {e}")
            return False

    def run(self, max_iterations=None):
        """Основной цикл работы"""
        print("\n" + "=" * 50)
        print("🚀 ЗАПУСК COLAB WORKER")
        print("=" * 50)

        # Настройка
        self.setup()

        iteration = 0

        while True:
            iteration += 1

            if max_iterations and iteration > max_iterations:
                print(f"🛑 Достигнут лимит итераций: {max_iterations}")
                break

            print(f"\n🔄 Итерация #{iteration}")

            # Проверяем задания
            job = self.check_jobs()

            if job:
                # Обрабатываем задание
                self.process_job(job)
                print("✅ Задание обработано")
            else:
                print(f"ℹ️  Заданий нет, ожидаем {self.check_interval} сек...")
                time.sleep(self.check_interval)

            # Для демо - ограничим количество итераций
            if iteration >= 3:
                print("\n🎯 Демо завершено! 3 итерации выполнены.")
                print("Для реальной работы установите max_iterations=None")
                break


# Простая функция для теста
def test_worker():
    """Тест работы воркера"""
    print("🧪 Тест Colab Worker...")

    worker = ColabWorker(check_interval=10)

    # Создаем тестовое задание
    repo_dir = Path("/content/focus-vae-experiment")
    pending_dir = repo_dir / "experiments" / "jobs" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    test_job = {
        "id": f"test_{int(time.time())}",
        "name": "Test Job",
        "description": "Тестовое задание для Colab Worker",
        "created_at": datetime.now().isoformat(),
        "status": "pending"
    }

    job_file = pending_dir / f"{test_job['id']}.json"
    with open(job_file, 'w') as f:
        json.dump(test_job, f, indent=2)

    print(f"✅ Тестовое задание создано: {job_file}")

    # Запускаем воркер на 1 итерацию
    worker.run(max_iterations=1)

    print("\n✅ Тест завершен!")