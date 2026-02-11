#!/usr/bin/env python3
"""
Colab Setup - полная настройка окружения одной командой
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("🚀 VAE EXPERIMENT SYSTEM - НАСТРОЙКА COLAB")
    print("=" * 60)

    # 1. Проверка Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Запущено в Google Colab")
    except:
        IN_COLAB = False
        print("⚠️ Запущено не в Colab")

    # 2. GitHub Token
    if IN_COLAB:
        print("\n🔑 Проверка GitHub Token...")
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
            os.environ['GITHUB_TOKEN'] = token
            print("✅ GitHub Token найден")
        except:
            print("\n❌ GitHub Token не найден!")
            print("\n📌 НАСТРОЙТЕ ТОКЕН:")
            print("   1. Нажмите 🔑 в левой панели")
            print("   2. Secrets → + Add new secret")
            print("   3. Name: GITHUB_TOKEN")
            print("   4. Value: ваш_токен")
            print("   5. ☑️ Поставьте галочку")
            print("\n👉 После настройки ПЕРЕЗАПУСТИТЕ ЯЧЕЙКУ")
            return

    # 3. Установка зависимостей
    print("\n📦 Установка зависимостей...")
    deps = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "numpy matplotlib seaborn scikit-learn",
        "tqdm gitpython requests pillow ipywidgets"
    ]
    for dep in deps:
        os.system(f"pip install {dep} -q")
    print("✅ Зависимости установлены")

    # 4. Клонирование репозитория
    print("\n📥 Клонирование репозитория...")
    repo_url = "https://github.com/Alexeiyaganov/focus-vae-experiment.git"

    if IN_COLAB:
        repo_url = f"https://{token}@github.com/Alexeiyaganov/focus-vae-experiment.git"

    repo_path = Path("/content/focus-vae-experiment")

    if not repo_path.exists():
        os.system(f"git clone {repo_url} {repo_path}")
    else:
        os.chdir(repo_path)
        os.system("git pull")

    os.chdir(repo_path)
    print(f"✅ Репозиторий: {repo_path}")

    # 5. Создание структуры
    print("\n📁 Создание структуры...")
    folders = [
        "experiments/jobs/pending",
        "experiments/jobs/running",
        "experiments/jobs/completed",
        "experiments/jobs/failed",
        "experiments/results",
        "experiments/logs",
        "configs"
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {folder}")

    # 6. Проверка импортов
    print("\n🔍 Проверка импортов...")
    sys.path.append(str(repo_path))

    try:
        from scripts import create_job, worker, experiment_runner, github_connector
        print("✅ Все модули успешно импортированы")
    except Exception as e:
        print(f"⚠️ Ошибка импорта: {e}")

    # 7. Готово
    print("\n" + "=" * 60)
    print("✅ НАСТРОЙКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("""
📋 КОМАНДЫ ДЛЯ РАБОТЫ:

1. СОЗДАТЬ ЗАДАНИЕ:
   from scripts.create_job import create_quick_test
   job_id = create_quick_test()

2. ЗАПУСТИТЬ ВОРКЕР:
   from scripts.worker import start_worker
   start_worker(check_interval=60, max_jobs=10)

3. ПОСМОТРЕТЬ РЕЗУЛЬТАТЫ:
   !ls experiments/results/
   !cat experiments/results/ID/results.json

4. СОЗДАТЬ ПАКЕТ ЗАДАНИЙ:
   from scripts.create_job import JobCreator
   creator = JobCreator()
   creator.create_batch_jobs([
       ("quick_test", None, 1),
       ("full_comparison", {"epochs": 10}, 2)
   ])
""")


if __name__ == "__main__":
    main()