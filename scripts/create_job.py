"""
Colab Worker - выполняет задания из очереди
"""

import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch

class ColabWorker:
    """Выполнитель заданий в Colab"""

    def __init__(self,
                 repo_path: str = "/content/focus-vae-experiment",
                 check_interval: int = 60):

        self.repo_path = Path(repo_path)
        self.pending_dir = self.repo_path / "experiments" / "jobs" / "pending"
        self.running_dir = self.repo_path / "experiments" / "jobs" / "running"
        self.completed_dir = self.repo_path / "experiments" / "jobs" / "completed"
        self.failed_dir = self.repo_path / "experiments" / "jobs" / "failed"
        self.results_dir = self.repo_path / "experiments" / "results"

        self.check_interval = check_interval
        self.current_job = None

        # Создаем папки
        for dir_path in [self.pending_dir, self.running_dir,
                        self.completed_dir, self.failed_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Информация о системе
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_info = self._get_gpu_info()

        print(f"👷 Colab Worker инициализирован")
        print(f"   📁 Репозиторий: {self.repo_path}")
        print(f"   💻 Устройство: {self.device}")
        if self.gpu_info:
            print(f"   🎮 GPU: {self.gpu_info}")
        print(f"   ⏱️  Интервал проверки: {check_interval}с")

    def _get_gpu_info(self) -> Optional[str]:
        """Получение информации о GPU"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Получение следующего задания из очереди"""
        job_files = list(self.pending_dir.glob("*.json"))

        if not job_files:
            return None

        # Сортируем по приоритету (читаем из файла)
        jobs = []
        for job_file in job_files:
            try:
                with open(job_file, 'r') as f:
                    job = json.load(f)
                jobs.append((job_file, job))
            except:
                continue

        # Сортируем по приоритету (меньше = выше)
        jobs.sort(key=lambda x: x[1].get('priority', 5))

        if jobs:
            job_file, job = jobs[0]

            # Перемещаем в running
            running_file = self.running_dir / job_file.name
            job_file.rename(running_file)

            job['status'] = 'running'
            job['started_at'] = datetime.now().isoformat()
            job['device'] = self.device

            with open(running_file, 'w') as f:
                json.dump(job, f, indent=2)

            self.current_job = job
            return job

        return None

    def run_experiment(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск эксперимента"""
        print(f"\n🚀 Запуск эксперимента: {job['id']}")
        print(f"   📊 Конфиг: {job['config_name']}")
        print(f"   🤖 Модели: {job['config']['models']}")

        try:
            # Импортируем раннер
            sys.path.append(str(self.repo_path))
            from scripts.experiment_runner import run_experiment

            # Создаем папку для результатов
            result_dir = self.results_dir / job['id']
            result_dir.mkdir(parents=True, exist_ok=True)

            # Запускаем эксперимент
            start_time = time.time()
            results = run_experiment(job['config'], result_dir)
            end_time = time.time()

            # Добавляем метаданные
            results.update({
                'job_id': job['id'],
                'device': self.device,
                'gpu': self.gpu_info,
                'runtime_seconds': end_time - start_time,
                'completed_at': datetime.now().isoformat()
            })

            # Сохраняем результаты
            with open(result_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)

            print(f"✅ Эксперимент завершен за {results['runtime_seconds']:.1f}с")
            return results

        except Exception as e:
            print(f"❌ Ошибка выполнения: {e}")
            import traceback
            traceback.print_exc()

            return {
                'job_id': job['id'],
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def complete_job(self, job: Dict[str, Any], results: Dict[str, Any]):
        """Завершение задания"""
        running_file = self.running_dir / f"{job['id']}.json"

        if results.get('status') == 'failed' or 'error' in results:
            # Перемещаем в failed
            dest_dir = self.failed_dir
            job['status'] = 'failed'
        else:
            # Перемещаем в completed
            dest_dir = self.completed_dir
            job['status'] = 'completed'

        job['completed_at'] = datetime.now().isoformat()
        job['results'] = results

        dest_file = dest_dir / f"{job['id']}.json"

        if running_file.exists():
            with open(dest_file, 'w') as f:
                json.dump(job, f, indent=2)
            running_file.unlink()

        print(f"📌 Задание {job['id']} -> {job['status']}")

    def push_to_github(self):
        """Отправка результатов в GitHub"""
        try:
            from scripts.github_connector import GitHubConnector
            connector = GitHubConnector()

            commit_msg = f"Colab: Результаты {self.current_job['id'] if self.current_job else 'batch'}"
            if connector.push_results(commit_msg):
                print("✅ Результаты отправлены в GitHub")
            else:
                print("⚠️ Не удалось отправить в GitHub")
        except Exception as e:
            print(f"⚠️ Ошибка GitHub: {e}")

    def run(self, max_jobs: Optional[int] = None, push_to_github: bool = True):
        """Основной цикл работы"""
        print("\n" + "=" * 60)
        print("🚀 ЗАПУСК COLAB WORKER")
        print("=" * 60)

        jobs_processed = 0

        try:
            while True:
                # Проверяем лимит заданий
                if max_jobs and jobs_processed >= max_jobs:
                    print(f"\n🛑 Достигнут лимит заданий: {max_jobs}")
                    break

                # Получаем следующее задание
                job = self.get_next_job()

                if job:
                    jobs_processed += 1
                    print(f"\n📋 Задание #{jobs_processed}")

                    # Выполняем эксперимент
                    results = self.run_experiment(job)

                    # Завершаем задание
                    self.complete_job(job, results)

                    # Отправляем в GitHub
                    if push_to_github:
                        self.push_to_github()

                    print(f"\n✅ Задание {job['id']} обработано")
                else:
                    print(f"\n⏳ [{datetime.now().strftime('%H:%M:%S')}] Нет заданий, жду {self.check_interval}с...")
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Работник остановлен пользователем")

        print(f"\n📊 Итого обработано заданий: {jobs_processed}")
        return jobs_processed


# Функция для быстрого запуска
def start_worker(check_interval: int = 60, max_jobs: Optional[int] = None):
    """Быстрый запуск работника"""
    worker = ColabWorker(check_interval=check_interval)
    return worker.run(max_jobs=max_jobs)


if __name__ == "__main__":
    # Запуск в демо-режиме
    worker = ColabWorker(check_interval=30)
    worker.run(max_jobs=2)  # Обработать 2 задания и остановиться