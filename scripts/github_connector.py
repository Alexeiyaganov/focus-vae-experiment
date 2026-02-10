"""
Коннектор для работы с GitHub из Colab
"""

import os
import subprocess
import json
from pathlib import Path
from google.colab import userdata


class GitHubConnector:
    def __init__(self, repo_owner="Alexeiyaganov", repo_name="focus-vae-experiment"):
        """
        Инициализация коннектора к GitHub

        Args:
            repo_owner: Владелец репозитория
            repo_name: Название репозитория
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name

        # Получаем токен из Colab Secrets
        try:
            self.gh_token = userdata.get('GITHUB_TOKEN')
            print("✅ GitHub токен загружен из Colab Secrets")
        except Exception as e:
            print(f"❌ Ошибка загрузки токена: {e}")
            print("\n🔧 Как настроить:")
            print("1. Нажмите на значок 🔑 слева в Colab")
            print("2. Выберите 'Secrets' (NOTA BENE)")
            print("3. Добавьте новый секрет:")
            print("   Имя: GITHUB_TOKEN")
            print("   Значение: ваш_github_token")
            raise

        # URL репозитория с токеном
        self.repo_url = f"https://{self.gh_token}@github.com/{repo_owner}/{repo_name}.git"

        # Пути
        self.base_dir = Path("/content")
        self.repo_dir = self.base_dir / repo_name

        print(f"📁 Репозиторий: {repo_owner}/{repo_name}")
        print(f"📁 Локальная папка: {self.repo_dir}")

    def setup_git_config(self):
        """Настройка git конфигурации"""
        print("🔧 Настраиваем git...")

        # Устанавливаем глобальные настройки git
        subprocess.run(["git", "config", "--global", "user.name", "Colab Worker"],
                       capture_output=True, text=True)
        subprocess.run(["git", "config", "--global", "user.email", "colab@worker.com"],
                       capture_output=True, text=True)

        # Сохраняем credentials для автоматической аутентификации
        credentials_file = Path.home() / ".git-credentials"
        with open(credentials_file, "w") as f:
            f.write(f"https://{self.gh_token}:x-oauth-basic@github.com\n")

        subprocess.run(["git", "config", "--global", "credential.helper", "store"],
                       capture_output=True, text=True)

        print("✅ Git настроен")

    def clone_or_pull_repository(self):
        """
        Клонирует или обновляет репозиторий

        Returns:
            bool: Успешно ли выполнена операция
        """
        print("🔄 Работа с репозиторием...")

        try:
            # Переходим в базовую директорию
            os.chdir(self.base_dir)

            if self.repo_dir.exists():
                # Репозиторий уже существует, обновляем
                print("📂 Репозиторий уже существует, обновляем...")
                os.chdir(self.repo_dir)

                # Сохраняем локальные изменения если есть
                result = subprocess.run(["git", "stash"], capture_output=True, text=True)
                if "Saved" in result.stdout:
                    print("💾 Локальные изменения сохранены в stash")

                # Обновляем из origin
                result = subprocess.run(["git", "pull", "origin", "main", "--force"],
                                        capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"⚠️ Ошибка при pull: {result.stderr}")
                    # Пробуем полный ресет
                    subprocess.run(["git", "reset", "--hard", "origin/main"],
                                   capture_output=True, text=True)

                print("✅ Репозиторий обновлен")

            else:
                # Клонируем репозиторий
                print("📥 Клонируем репозиторий...")
                result = subprocess.run(
                    ["git", "clone", self.repo_url, self.repo_name],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"❌ Ошибка клонирования: {result.stderr}")
                    return False

                os.chdir(self.repo_dir)
                print("✅ Репозиторий клонирован")

            # Проверяем, что мы в правильной ветке
            subprocess.run(["git", "checkout", "main"], capture_output=True, text=True)

            return True

        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            return False

    def install_dependencies(self):
        """Установка зависимостей проекта"""
        print("📦 Устанавливаем зависимости...")

        requirements_file = self.repo_dir / "requirements.txt"

        if not requirements_file.exists():
            print("⚠️ Файл requirements.txt не найден, устанавливаем базовые...")
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
                "Pillow>=9.5.0"
            ]

            with open(requirements_file, "w") as f:
                f.write("\n".join(dependencies))

        # Устанавливаем зависимости
        result = subprocess.run(
            ["pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ Зависимости установлены")
        else:
            print(f"⚠️ Предупреждение при установке: {result.stderr}")

    def push_results(self, commit_message="Colab: Experiment results"):
        """
        Пушит изменения в GitHub

        Args:
            commit_message: Сообщение коммита
        """
        print("💾 Сохраняем изменения в GitHub...")

        try:
            # Проверяем, есть ли изменения
            result = subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True)

            if not result.stdout.strip():
                print("ℹ️ Нет изменений для коммита")
                return True

            # Добавляем все изменения
            subprocess.run(["git", "add", "-A"], capture_output=True, text=True)

            # Коммитим
            subprocess.run(["git", "commit", "-m", commit_message],
                           capture_output=True, text=True)

            # Пушим
            result = subprocess.run(["git", "push", "origin", "main"],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Изменения успешно запушены в GitHub")
                return True
            else:
                print(f"❌ Ошибка при пуше: {result.stderr}")

                # Пробуем с force если нужно
                print("🔄 Пробуем force push...")
                result = subprocess.run(["git", "push", "origin", "main", "--force"],
                                        capture_output=True, text=True)

                if result.returncode == 0:
                    print("✅ Force push успешен")
                    return True
                else:
                    print(f"❌ Force push тоже не удался: {result.stderr}")
                    return False

        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
            return False

    def get_repository_info(self):
        """Получение информации о репозитории"""
        print("📊 Информация о репозитории:")

        # Текущая ветка
        result = subprocess.run(["git", "branch", "--show-current"],
                                capture_output=True, text=True)
        print(f"  Ветка: {result.stdout.strip()}")

        # Последний коммит
        result = subprocess.run(["git", "log", "-1", "--oneline"],
                                capture_output=True, text=True)
        print(f"  Последний коммит: {result.stdout.strip()}")

        # Статус
        result = subprocess.run(["git", "status", "--short"],
                                capture_output=True, text=True)
        changes = len([line for line in result.stdout.strip().split('\n') if line])
        print(f"  Изменений: {changes}")


# Пример использования
def setup_colab_environment():
    """Настройка окружения Colab за один вызов"""
    print("=" * 60)
    print("🚀 НАСТРОЙКА COLAB + GITHUB")
    print("=" * 60)

    # Создаем коннектор
    connector = GitHubConnector()

    # Настраиваем git
    connector.setup_git_config()

    # Клонируем/обновляем репозиторий
    if not connector.clone_or_pull_repository():
        print("❌ Не удалось настроить репозиторий")
        return None

    # Устанавливаем зависимости
    connector.install_dependencies()

    # Показываем информацию
    connector.get_repository_info()

    print("=" * 60)
    print("✅ ОКРУЖЕНИЕ НАСТРОЕНО!")
    print("=" * 60)

    return connector