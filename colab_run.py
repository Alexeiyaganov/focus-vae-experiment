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
    'models': ['vae', 'iwae', 'focus_vae']  # ВСЕ 4 МОДЕЛИ
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

# ========== 7. СОЗДАНИЕ ГРАФИКОВ ==========
print("\n" + "=" * 60)
print("📊 СОЗДАНИЕ ГРАФИКОВ")
print("=" * 60)

try:
    import matplotlib.pyplot as plt

    # График сходимости
    plt.figure(figsize=(14, 7))

    # Цвета для разных моделей
    colors = {
        'vae': 'blue',
        'iwae': 'orange',
        'vamp': 'green',
        'focus_vae': 'red'
    }

    for model_name, model_results in results['models'].items():
        losses = model_results.get('train_losses', [])
        if losses:
            plt.plot(losses,
                    label=model_name.upper(),
                    color=colors.get(model_name, 'gray'),
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    markevery=max(1, len(losses)//5))

    plt.xlabel('Эпоха', fontsize=14, fontweight='bold')
    plt.ylabel('Loss (ELBO)', fontsize=14, fontweight='bold')
    plt.title('Сравнение скорости сходимости моделей VAE', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Сохраняем локально
    plots_dir = Path('experiments/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / 'convergence_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"   ✅ График сходимости создан: {plot_path}")

    # Добавляем путь к графику в результаты для GitHub
    results['plots'] = {
        'convergence': str(plot_path)
    }

except Exception as e:
    print(f"   ⚠️ Ошибка создания графиков: {e}")
    import traceback
    traceback.print_exc()

# ========== 8. СОХРАНЕНИЕ В GITHUB ==========
if TOKEN:
    print("\n" + "=" * 60)
    print("📤 СОХРАНЕНИЕ В GITHUB")
    print("=" * 60)

    try:
        # Передаем результаты и графики
        save_to_github(TOKEN, results)
        print("   ✅ Результаты сохранены в GitHub")
    except Exception as e:
        print(f"   ❌ Ошибка при сохранении: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️ Результаты не сохранены в GitHub")
    print("   Добавьте GITHUB_TOKEN в Secrets Colab для автоматического сохранения")

# ========== 9. ВЫВОД РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 60)
print("📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
print("=" * 60)

print("\n   " + "-" * 50)
print("   {:<12} | {:>12} | {:>12}".format("Модель", "Train Loss", "Test Loss"))
print("   " + "-" * 50)

for model_name in ['vae', 'iwae', 'vamp', 'focus_vae']:
    model_results = results['models'].get(model_name, {})
    train_loss = model_results.get('final_train_loss', 0)
    test_loss = model_results.get('test_loss', 0)

    # Определяем победителя (минимальный тестовый лосс)
    winner = " 🏆" if test_loss == min([m.get('test_loss', float('inf'))
                                       for m in results['models'].values()]) else ""

    print(f"   {model_name.upper():<12} | {train_loss:>12.2f} | {test_loss:>12.2f}{winner}")

print("   " + "-" * 50)

# Лучшая модель
best_model = min(results['models'].items(), key=lambda x: x[1].get('test_loss', float('inf')))
print(f"\n🏆 Лучшая модель: {best_model[0].upper()} (Test Loss: {best_model[1].get('test_loss', 0):.2f})")

print("\n" + "=" * 60)
print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("=" * 60)
print(f"\n📁 Результаты сохранены локально в: {repo_path}/experiments/")
if TOKEN:
    print(f"📤 Результаты отправлены в GitHub: {TOKEN[:4]}...{TOKEN[-4:]}")
print("=" * 60)