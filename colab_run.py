"""
ОДНА КОМАНДА для запуска в Colab
Скачивает, обучает, сохраняет в GitHub
"""

import os
import sys
import subprocess
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
        subprocess.run(f"git clone https://{TOKEN}@github.com/Alexeiyaganov/focus-vae-experiment.git",
                      shell=True, check=True)
    else:
        subprocess.run("git clone https://github.com/Alexeiyaganov/focus-vae-experiment.git",
                      shell=True, check=True)

os.chdir(repo_path)
print(f"   ✅ Репозиторий: {repo_path}")

# ========== 3. УСТАНОВКА ==========
print("\n📦 Установка зависимостей...")
subprocess.run("pip install torch torchvision numpy matplotlib requests -q", shell=True)
print("   ✅ Готово")

# ========== 4. ИМПОРТ МОДУЛЕЙ ==========
print("\n🔧 Загрузка модулей...")
sys.path.append(str(repo_path))

try:
    from train import run_experiment
    from results import save_to_github
    print("   ✅ Модули загружены")
except ImportError as e:
    print(f"   ❌ Ошибка загрузки модулей: {e}")
    print("   Проверьте, что файлы train.py и results.py есть в репозитории")
    sys.exit(1)

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
print(f"   Batch size: {config['batch_size']}")
print(f"   Latent dim: {config['latent_dim']}")
print(f"   Модели: {config['models']}")

# ========== 6. ЗАПУСК ==========
print("\n" + "=" * 60)
print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА")
print("=" * 60)

try:
    results = run_experiment(config)
    print("   ✅ Эксперимент выполнен успешно")
except Exception as e:
    print(f"   ❌ Ошибка при выполнении эксперимента: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 7. СОХРАНЕНИЕ ==========
if TOKEN:
    print("\n" + "=" * 60)
    print("📤 СОХРАНЕНИЕ В GITHUB")
    print("=" * 60)

    try:
        save_to_github(TOKEN, results)
        print("   ✅ Результаты сохранены в GitHub")
    except Exception as e:
        print(f"   ❌ Ошибка при сохранении: {e}")
else:
    print("\n⚠️ Результаты не сохранены в GitHub")
    print("   Добавьте GITHUB_TOKEN в Secrets Colab для автоматического сохранения")

# ========== 8. ВЫВОД РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 60)
print("📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
print("=" * 60)

vae_results = results['models'].get('vae', {})
focus_results = results['models'].get('focus_vae', {})

print(f"""
   VAE:       Train Loss: {vae_results.get('final_train_loss', 0):.2f}
              Test Loss:  {vae_results.get('test_loss', 0):.2f}
   
   FocusVAE:  Train Loss: {focus_results.get('final_train_loss', 0):.2f}
              Test Loss:  {focus_results.get('test_loss', 0):.2f}
""")

print("=" * 60)
print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("=" * 60)