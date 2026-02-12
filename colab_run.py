"""
ОДНА КОМАНДА для запуска в Colab
Скачивает, обучает, сохраняет в GitHub
"""

import os
import sys
from pathlib import Path

# ========== 1. ПРОВЕРКА TOKEN ==========
print("\n🔑 GitHub Token:")
try:
    from google.colab import userdata

    TOKEN = userdata.get('GITHUB_TOKEN')
    print("   ✅ Токен найден")
except:
    TOKEN = None
    print("   ⚠️ Токен не найден, результаты не сохранятся в GitHub")
    print("     (Добавьте GITHUB_TOKEN в Secrets Colab для сохранения)")

# ========== 2. КЛОНИРОВАНИЕ ==========
print("\n📥 Клонирование репозитория...")
repo_path = Path("/content/focus-vae-experiment")

if not repo_path.exists():
    if TOKEN:
        !git
        clone
        https: // {TOKEN} @ github.com / Alexeiyaganov / focus - vae - experiment.git
    else:
        !git
        clone
        https: // github.com / Alexeiyaganov / focus - vae - experiment.git

os.chdir(repo_path)
print(f"   ✅ Репозиторий: {repo_path}")

# ========== 3. УСТАНОВКА ==========
print("\n📦 Установка зависимостей...")
!pip
install
torch
torchvision
numpy
matplotlib
requests - q
print("   ✅ Готово")

# ========== 4. ИМПОРТ МОДУЛЕЙ ==========
print("\n🔧 Загрузка модулей...")
sys.path.append(str(repo_path))

from train import run_experiment
from results import save_to_github

# ========== 5. КОНФИГУРАЦИЯ ==========
config = {
    'epochs': 5,  # Быстрый тест
    'batch_size': 64,
    'latent_dim': 20,
    'learning_rate': 3e-4,
    'models': ['vae', 'focus_vae']
}

print(f"\n⚙️ Конфигурация:")
print(f"   Эпохи: {config['epochs']}")
print(f"   Модели: {config['models']}")

# ========== 6. ЗАПУСК ==========
print("\n" + "=" * 60)
print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА")
print("=" * 60)

results = run_experiment(config)

# ========== 7. СОХРАНЕНИЕ ==========
if TOKEN:
    print("\n" + "=" * 60)
    print("📤 СОХРАНЕНИЕ В GITHUB")
    print("=" * 60)

    save_to_github(TOKEN, results)
else:
    print("\n⚠️ Результаты не сохранены в GitHub")
    print("   Добавьте GITHUB_TOKEN в Secrets Colab и перезапустите")

print("\n" + "=" * 60)
print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("=" * 60)
print(f"""
📊 РЕЗУЛЬТАТЫ:

   VAE:       Train Loss: {results['models']['vae']['final_train_loss']:.2f}
              Test Loss:  {results['models']['vae']['test_loss']:.2f}

   FocusVAE:  Train Loss: {results['models']['focus_vae']['final_train_loss']:.2f}
              Test Loss:  {results['models']['focus_vae']['test_loss']:.2f}
""")