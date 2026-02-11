"""
Colab Setup Script - Однокнопочная настройка
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🚀 НАСТРОЙКА COLAB ДЛЯ VAE ЭКСПЕРИМЕНТОВ")
print("=" * 60)

# 1. Проверка Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("✅ Запущено в Google Colab")
except:
    IN_COLAB = False
    print("⚠️  Запущено не в Colab, некоторые функции недоступны")

# 2. Установка зависимостей
print("\n📦 Шаг 1: Установка зависимостей...")

dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.65.0",
    "gitpython>=3.1.0",
    "requests>=2.31.0",
    "Pillow>=9.5.0",
    "ipywidgets>=8.0.0"
]

for dep in dependencies:
    package = dep.split(">=")[0].split("[")[0]
    print(f"  Устанавливаем {package}...")
    os.system(f"pip install {dep} -q")

print("✅ Зависимости установлены")

# 3. Настройка GitHub Token
print("\n🔑 Шаг 2: Настройка GitHub Token...")

if IN_COLAB:
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
        print("✅ GitHub токен найден в Colab Secrets")

        # Сохраняем credentials
        creds_file = Path.home() / ".git-credentials"
        with open(creds_file, "w") as f:
            f.write(f"https://{token}:x-oauth-basic@github.com\n")

        print("✅ Git credentials сохранены")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n" + "=" * 50)
        print("⚠️  НАСТРОЙТЕ GITHUB TOKEN:")
        print("=" * 50)
        print("""
1. Нажмите на значок 🔑 слева в Colab
2. Выберите вкладку 'Secrets (NOTA BENE)'
3. Нажмите '+ Add new secret'
4. Введите:
   Name: GITHUB_TOKEN
   Value: ваш_github_personal_token

КАК ПОЛУЧИТЬ ТОКЕН:
1. Зайдите на GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token (classic)
4. Выберите scopes: repo (полный доступ)
5. Скопируйте токен и вставьте в Colab Secrets
        """)
        sys.exit(1)
else:
    print("ℹ️  Запущено вне Colab, используйте локальный токен")

# 4. Клонирование репозитория
print("\n📥 Шаг 3: Клонирование репозитория...")

repo_url = "https://github.com/Alexeiyaganov/focus-vae-experiment.git"
repo_dir = Path("/content/focus-vae-experiment")

if not repo_dir.exists():
    print(f"  Клонируем {repo_url}...")

    # Используем токен если в Colab
    if IN_COLAB:
        import subprocess
        result = subprocess.run(
            f"git clone https://{token}@github.com/Alexeiyaganov/focus-vae-experiment.git",
            shell=True,
            capture_output=True,
            text=True
        )
    else:
        os.system(f"git clone {repo_url}")

    if repo_dir.exists():
        print(f"✅ Репозиторий клонирован: {repo_dir}")
    else:
        print("❌ Не удалось клонировать репозиторий")
        sys.exit(1)
else:
    print(f"✅ Репозиторий уже существует: {repo_dir}")

# 5. Настройка рабочей директории
os.chdir(repo_dir)
print(f"📂 Рабочая директория: {os.getcwd()}")

# 6. Импорт системы
print("\n🔧 Шаг 4: Импорт системы...")

# Добавляем scripts в путь
sys.path.append(str(repo_dir / "scripts"))

try:
    from github_connector import GitHubConnector
    print("✅ GitHubConnector загружен")

    from colab_worker import ColabWorker
    print("✅ ColabWorker загружен")

    from experiment_runner import run_experiment
    print("✅ Experiment runner загружен")

    print("✅ Все модули успешно загружены")

except ImportError as e:
    print(f"⚠️  Ошибка импорта: {e}")
    print("Создаем базовые файлы...")

    # Создаем недостающие файлы
    create_missing_files(repo_dir)

    # Пробуем снова
    from github_connector import GitHubConnector
    from colab_worker import ColabWorker

# 7. Запуск тестового соединения
print("\n🔗 Шаг 5: Тест соединения с GitHub...")

try:
    connector = GitHubConnector()

    # Настраиваем git
    connector.setup_git_config()

    # Клонируем/обновляем
    if connector.clone_or_pull_repository():
        print("✅ Соединение с GitHub установлено")

        # Показываем информацию
        import subprocess
        result = subprocess.run(["git", "branch", "--show-current"],
                              capture_output=True, text=True)
        print(f"📌 Текущая ветка: {result.stdout.strip()}")

        result = subprocess.run(["git", "log", "-1", "--oneline"],
                              capture_output=True, text=True)
        print(f"📌 Последний коммит: {result.stdout.strip()}")
    else:
        print("⚠️  Проблемы с соединением GitHub")

except Exception as e:
    print(f"❌ Ошибка соединения: {e}")

# 8. Готово
print("\n" + "=" * 60)
print("✅ НАСТРОЙКА ЗАВЕРШЕНА!")
print("=" * 60)
print("""
🎯 ЧТО ДЕЛАТЬ ДАЛЬШЕ:

1. Запустить Colab Worker (выполняет задания):
   from scripts.colab_worker import ColabWorker
   worker = ColabWorker()
   worker.run()

2. Создать новое задание:
   from scripts.create_job import create_experiment_job
   job_id = create_experiment_job("quick_test", epochs=5)

3. Проверить результаты:
   !ls experiments/results/
   !ls experiments/jobs/completed/

📚 ДОКУМЕНТАЦИЯ:
- GitHub репозиторий: https://github.com/Alexeiyaganov/focus-vae-experiment
- Colab ноутбук: этот файл
""")
print("=" * 60)

# Функция для создания недостающих файлов
def create_missing_files(repo_dir):
    """Создает недостающие файлы если их нет"""

    scripts_dir = repo_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Создаем __init__.py
    init_file = scripts_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write("# Focus VAE Experiment Scripts\n")

    print(f"✅ Создана папка scripts: {scripts_dir}")