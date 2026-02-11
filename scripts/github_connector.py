"""
GitHub Connector для Colab
Простое подключение к GitHub репозиторию
"""

import os
import subprocess
import sys
from pathlib import Path

class GitHubConnector:
    def __init__(self, repo_owner="Alexeiyaganov", repo_name="focus-vae-experiment"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name

        # Пытаемся получить токен из разных источников
        self.gh_token = self._get_github_token()

        if self.gh_token:
            self.repo_url = f"https://{self.gh_token}@github.com/{repo_owner}/{repo_name}.git"
        else:
            self.repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"

        self.base_dir = Path("/content")
        self.repo_dir = self.base_dir / repo_name

    def _get_github_token(self):
        """Получение GitHub токена из разных источников"""
        # 1. Из Colab Secrets
        try:
            from google.colab import userdata
            return userdata.get('GITHUB_TOKEN')
        except:
            pass

        # 2. Из переменных окружения
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            return token

        # 3. Из файла credentials
        creds_file = Path.home() / ".git-credentials"
        if creds_file.exists():
            with open(creds_file, "r") as f:
                content = f.read()
                if "github.com" in content:
                    # Пытаемся извлечь токен
                    import re
                    match = re.search(r'https://([^:@]+):', content)
                    if match:
                        return match.group(1)

        print("⚠️  GitHub токен не найден. Будут доступны только публичные операции.")
        return None

    def setup_git_config(self):
        """Базовая настройка git"""
        subprocess.run(["git", "config", "--global", "user.name", "Colab Worker"],
                      capture_output=True)
        subprocess.run(["git", "config", "--global", "user.email", "colab@worker.com"],
                      capture_output=True)

        if self.gh_token:
            # Сохраняем credentials
            creds_file = Path.home() / ".git-credentials"
            with open(creds_file, "w") as f:
                f.write(f"https://{self.gh_token}:x-oauth-basic@github.com\n")

            subprocess.run(["git", "config", "--global", "credential.helper", "store"],
                          capture_output=True)

        return True

    def clone_or_pull_repository(self):
        """Клонирует или обновляет репозиторий"""
        try:
            os.chdir(self.base_dir)

            if self.repo_dir.exists():
                # Обновляем существующий
                os.chdir(self.repo_dir)

                # Сохраняем изменения
                subprocess.run(["git", "stash"], capture_output=True)

                # Обновляем
                result = subprocess.run(["git", "pull", "origin", "main"],
                                       capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"⚠️  Ошибка при pull: {result.stderr[:200]}")
                    return False

                print(f"✅ Репозиторий обновлен: {self.repo_dir}")

            else:
                # Клонируем новый
                print(f"📥 Клонируем репозиторий: {self.repo_url}")

                result = subprocess.run(
                    ["git", "clone", self.repo_url, self.repo_name],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"❌ Ошибка клонирования: {result.stderr[:200]}")
                    return False

                os.chdir(self.repo_dir)
                print(f"✅ Репозиторий клонирован: {self.repo_dir}")

            # Проверяем ветку
            subprocess.run(["git", "checkout", "main"], capture_output=True)

            return True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    def push_changes(self, commit_message="Colab: auto commit"):
        """Пушит изменения в репозиторий"""
        try:
            # Проверяем изменения
            result = subprocess.run(["git", "status", "--porcelain"],
                                  capture_output=True, text=True)

            if not result.stdout.strip():
                print("ℹ️  Нет изменений для коммита")
                return True

            # Добавляем все
            subprocess.run(["git", "add", "-A"], capture_output=True)

            # Коммитим
            subprocess.run(["git", "commit", "-m", commit_message], capture_output=True)

            # Пушим
            result = subprocess.run(["git", "push", "origin", "main"],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Изменения запушены в GitHub")
                return True
            else:
                print(f"❌ Ошибка при пуше: {result.stderr[:200]}")
                return False

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

# Простая функция для быстрой настройки
def setup_colab_github():
    """Быстрая настройка Colab + GitHub"""
    print("🔧 Настройка Colab + GitHub...")

    connector = GitHubConnector()

    # Настраиваем git
    connector.setup_git_config()

    # Клонируем/обновляем
    success = connector.clone_or_pull_repository()

    if success:
        print("✅ Настройка завершена успешно!")
        return connector
    else:
        print("❌ Настройка не удалась")
        return None