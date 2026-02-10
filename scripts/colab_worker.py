"""
Полный Colab Worker с GitHub интеграцией
"""

import time
import json
import sys
from pathlib import Path

# Добавляем путь к нашим скриптам
sys.path.append('/content/focus-vae-experiment/scripts')

from github_connector import GitHubConnector, setup_colab_environment

class ColabWorkerComplete:
    def __init__(self, check_interval=300):
        """
        Инициализация Colab Worker

        Args:
            check_interval: Интервал проверки заданий в секундах
        """
        self.check_interval = check_interval
        self.connector = None
        self.repo_dir = None

        print("👷 Colab Worker с GitHub интеграцией")
        print(f"⏱️  Интервал проверки: {check_interval} секунд")

    def setup(self):
        """Настройка окружения"""
        print("\n🔧 Настраиваем окружение...")

        # Настраиваем Colab + GitHub
        self.connector = setup_colab_environment()
        if not self.connector:
            return False

        self.repo_dir = self.connector.repo_dir

        # Добавляем путь к скриптам для импорта
        sys.path.append(str(self.repo_dir / "scripts"))

        print("✅ Окружение настроено")
        return True

    def check_jobs(self):
        """Проверка новых заданий"""
        print("\n🔍 Проверяем задания...")

        # Путь к папке с заданиями
        jobs_pending_dir = self.repo_dir / "experiments" / "jobs" / "pending"

        if not jobs_pending_dir.exists():
            print("ℹ️  Папка заданий не найдена, создаём...")
            jobs_pending_dir.mkdir(parents=True, exist_ok=True)
            return None

        # Ищем JSON файлы с заданиями
        job_files = list(jobs_pending_dir.glob("*.json"))

        if not job_files:
            print("ℹ️  Нет ожидающих заданий")
            return None

        # Сортируем по времени создания
        job_files.sort(key=lambda x: x.stat().st_mtime)

        # Берем самое старое задание
        job_file = job_files[0]

        try:
            with open(job_file, 'r') as f:
                job = json.load(f)

            print(f"🎯 Найдено задание: {job.get('id', 'unknown')}")
            print(f"   Название: {job.get('name', 'No name')}")
            print(f"   Модели: {job.get('models', [])}")

            return job

        except Exception as e:
            print(f"❌ Ошибка чтения задания {job_file}: {e}")
            return None

    def move_job_to_running(self, job):
        """Перемещает задание в running"""
        job_id = job['id']

        pending_file = self.repo_dir / "experiments" / "jobs" / "pending" / f"{job_id}.json"
        running_file = self.repo_dir / "experiments" / "jobs" / "running" / f"{job_id}.json"

        # Создаем папку running если нужно
        running_file.parent.mkdir(parents=True, exist_ok=True)

        # Обновляем статус
        job['status'] = 'running'
        job['started_at'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Сохраняем в running
        with open(running_file, 'w') as f:
            json.dump(job, f, indent=2)

        # Удаляем из pending
        if pending_file.exists():
            pending_file.unlink()

        print(f"📌 Задание {job_id} перемещено в running")

    def execute_experiment(self, job):
        """Выполнение эксперимента"""
        job_id = job['id']
        print(f"\n🚀 Начинаем выполнение задания: {job_id}")

        try:
            # Создаем директорию для результатов
            results_dir = self.repo_dir / "experiments" / "results" / job_id
            results_dir.mkdir(parents=True, exist_ok=True)

            # Сохраняем конфиг задания
            config_file = results_dir / "job_config.json"
            with open(config_file, 'w') as f:
                json.dump(job, f, indent=2)

            # Импортируем и запускаем эксперимент
            from experiment_runner import run_experiment
            results = run_experiment(job, results_dir)

            # Обновляем задание
            job['status'] = 'completed'
            job['completed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            job['results_summary'] = {
                'success': True,
                'final_loss': results.get('final_losses', {}),
                'training_time': results.get('training_time', 0)
            }

            # Сохраняем результаты
            results_file = results_dir / "experiment_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"✅ Эксперимент {job_id} выполнен успешно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка выполнения эксперимента: {e}")
            import traceback
            traceback.print_exc()

            # Сохраняем информацию об ошибке
            error_dir = self.repo_dir / "experiments" / "results" / f"{job_id}_error"
            error_dir.mkdir(parents=True, exist_ok=True)

            error_info = {
                'job_id': job_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(error_dir / "error.json", 'w') as f:
                json.dump(error_info, f, indent=2)

            return False

    def finish_job(self, job, success=True):
        """Завершение задания"""
        job_id = job['id']

        # Определяем папку назначения
        if success:
            dest_dir = "completed"
            status = "completed"
        else:
            dest_dir = "failed"
            status = "failed"

        source_file = self.repo_dir / "experiments" / "jobs" / "running" / f"{job_id}.json"
        dest_file = self.repo_dir / "experiments" / "jobs" / dest_dir / f"{job_id}.json"

        # Создаем папку назначения
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Обновляем статус
        job['status'] = status

        # Сохраняем в новую папку
        with open(dest_file, 'w') as f:
            json.dump(job, f, indent=2)

        # Удаляем из running
        if source_file.exists():
            source_file.unlink()

        print(f"📌 Задание {job_id} перемещено в {dest_dir}")

    def save_to_github(self, job_id):
        """Сохранение результатов в GitHub"""
        print("\n💾 Сохраняем результаты в GitHub...")

        commit_message = f"Colab: Результаты эксперимента {job_id}"

        if self.connector.push_results(commit_message):
            print("✅ Результаты успешно сохранены в GitHub")
            return True
        else:
            print("⚠️ Не удалось сохранить в GitHub, продолжаем локально")
            return False

    def run(self):
        """Основной цикл работы воркера"""
        print("\n" + "=" * 60)
        print("🚀 ЗАПУСК COLAB WORKER")
        print("=" * 60)

        # Настраиваем окружение
        if not self.setup():
            print("❌ Не удалось настроить окружение")
            return

        cycle_count = 0

        while True:
            cycle_count += 1
            print(f"\n🔄 Цикл #{cycle_count}")

            try:
                # Проверяем новые задания
                job = self.check_jobs()

                if job:
                    # Перемещаем задание в running
                    self.move_job_to_running(job)

                    # Выполняем эксперимент
                    success = self.execute_experiment(job)

                    # Завершаем задание
                    self.finish_job(job, success)

                    # Сохраняем в GitHub
                    self.save_to_github(job['id'])

                    print(f"🎉 Задание {job['id']} полностью обработано")

                else:
                    # Нет заданий, ждем
                    print(f"⏳ Ожидаем {self.check_interval} секунд...")
                    time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\n🛑 Работа остановлена пользователем")
                break

            except Exception as e:
                print(f"⚠️ Ошибка в основном цикле: {e}")
                import traceback
                traceback.print_exc()

                # Ждем перед повторной попыткой
                time.sleep(60)

# Точка входа
if __name__ == "__main__":
    # Создаем и запускаем воркер
    worker = ColabWorkerComplete(check_interval=300)  # 5 минут
    worker.run()