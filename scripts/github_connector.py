"""
GitHub Connector - синхронизация с GitHub
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

class GitHubConnector:
    """Подключение и работа с GitHub репозиторием"""

    def __init__(self,
                 repo_owner: str = "Alexeiyaganov",
                 repo_name: str = "focus-vae-experiment",
                 repo_path: str = "/content/focus-vae-experiment"):

        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_path = Path(repo_path)

        # Получаем токен из Colab Secrets
        self.token = self._get_github_token()

        if self.token:
            self.repo_url = f"https://{self.token}@github.com/{repo_owner}/{repo_name}.git"
        else:
            self.repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"

    def _get_github_token(self) -> Optional[str]:
        """Получение GitHub токена из Colab Secrets"""
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
            if token:
                print("✅ GitHub токен загружен")
                return token
        except:
            pass

        # Пробуем из переменных окружения
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            print("✅ GitHub токен из переменных окружения")
            return token

        print("⚠️ GitHub токен не найден")
        return None

    def setup_git(self):
        """Настройка git"""
        subprocess.run(["git", "config", "--global", "user.name", "Colab Worker"],
                      capture_output=True)
        subprocess.run(["git", "config", "--global", "user.email", "colab@worker.com"],
                      capture_output=True)

        if self.token:
            # Сохраняем credentials
            creds_file = Path.home() / ".git-credentials"
            with open(creds_file, "w") as f:
                f.write(f"https://{self.token}:x-oauth-basic@github.com\n")

            subprocess.run(["git", "config", "--global", "credential.helper", "store"],
                          capture_output=True)

    def clone_or_pull(self) -> bool:
        """Клонирование или обновление репозитория"""
        try:
            os.chdir("/content")

            if self.repo_path.exists():
                os.chdir(self.repo_path)
                subprocess.run(["git", "pull"], capture_output=True)
                print("✅ Репозиторий обновлен")
            else:
                result = subprocess.run(
                    ["git", "clone", self.repo_url, str(self.repo_path)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("✅ Репозиторий склонирован")
                else:
                    print(f"❌ Ошибка клонирования: {result.stderr}")
                    return False

            os.chdir(self.repo_path)
            return True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    def push_results(self, commit_message: str = "Colab: experiment results") -> bool:
        """Отправка результатов в GitHub"""
        try:
            os.chdir(self.repo_path)

            # Проверяем изменения
            result = subprocess.run(["git", "status", "--porcelain"],
                                  capture_output=True, text=True)

            if not result.stdout.strip():
                print("ℹ️ Нет изменений для коммита")
                return True

            # Добавляем все
            subprocess.run(["git", "add", "-A"], capture_output=True)

            # Коммитим
            subprocess.run(["git", "commit", "-m", commit_message],
                          capture_output=True)

            # Пушим
            result = subprocess.run(["git", "push", "origin", "main"],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Изменения отправлены в GitHub")
                return True
            else:
                print(f"❌ Ошибка push: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    def create_job_from_local(self, job_file: Path) -> bool:
        """Создание задания из локального файла"""
        try:
            os.chdir(self.repo_path)

            # Копируем файл в pending
            dest = self.repo_path / "experiments" / "jobs" / "pending" / job_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(job_file, dest)

            # Коммитим и пушим
            subprocess.run(["git", "add", "experiments/jobs/pending/"],
                          capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Add job: {job_file.name}"],
                          capture_output=True)
            subprocess.run(["git", "push"], capture_output=True)

            print(f"✅ Задание {job_file.name} добавлено в очередь")
            return True

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False