"""
Save Results to GitHub
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
import requests

class GitHubSaver:
    """Сохранение результатов в GitHub"""

    def __init__(self, token, repo_owner="Alexeiyaganov", repo_name="focus-vae-experiment"):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def save_file(self, path, content, commit_message):
        """Сохранить файл в GitHub"""
        url = f"{self.api_url}/{path}"

        # Кодируем содержимое
        if isinstance(content, str):
            encoded = base64.b64encode(content.encode()).decode()
        else:
            encoded = base64.b64encode(content).decode()

        # Сначала пытаемся получить файл (если существует)
        sha = None
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                sha = response.json().get('sha')
        except:
            pass

        # Данные для коммита
        data = {
            'message': commit_message,
            'content': encoded,
            'branch': 'main'
        }

        if sha:
            data['sha'] = sha

        # Отправляем
        response = requests.put(url, headers=self.headers, json=data)

        if response.status_code in [200, 201]:
            print(f"   ✅ {path}")
            return True
        else:
            print(f"   ❌ {path}: {response.status_code} - {response.text[:100]}")
            return False

    def save_experiment_results(self, experiment_id, results):
        """Сохранить все результаты эксперимента"""

        # Генерируем путь
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"experiments/results/{experiment_id}_{timestamp}"

        print(f"\n📤 Сохранение в GitHub: {base_path}")

        # Сохраняем JSON с результатами
        json_str = json.dumps(results, indent=2, default=str)
        self.save_file(
            f"{base_path}/results.json",
            json_str,
            f"Add experiment results: {experiment_id}"
        )

        # Сохраняем конфиг отдельно
        if 'config' in results:
            config_str = json.dumps(results['config'], indent=2, default=str)
            self.save_file(
                f"{base_path}/config.json",
                config_str,
                f"Add experiment config: {experiment_id}"
            )

        # Если есть графики, сохраняем их
        if 'plots' in results:
            plots = results['plots']
            for plot_name, plot_path in plots.items():
                if os.path.exists(plot_path):
                    try:
                        with open(plot_path, 'rb') as f:
                            plot_content = f.read()

                        # Определяем расширение файла
                        ext = os.path.splitext(plot_path)[1] or '.png'

                        self.save_file(
                            f"{base_path}/plots/{plot_name}{ext}",
                            plot_content,
                            f"Add plot: {plot_name}"
                        )
                        print(f"   ✅ График {plot_name} сохранен")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка сохранения графика {plot_name}: {e}")

        print(f"\n✅ Результаты сохранены в GitHub")
        print(f"   https://github.com/{self.repo_owner}/{self.repo_name}/tree/main/{base_path}")

        return base_path


def save_to_github(token, results):
    """Быстрое сохранение результатов"""
    saver = GitHubSaver(token)
    experiment_id = f"vae_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return saver.save_experiment_results(experiment_id, results)