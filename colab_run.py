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

# Сначала проверяем переменные окружения
TOKEN = os.environ.get('GITHUB_TOKEN')

# Если нет в окружении, пробуем из Colab Secrets
if not TOKEN:
    try:
        from google.colab import userdata
        TOKEN = userdata.get('GITHUB_TOKEN')
        print("   ✅ Токен найден в Colab Secrets")
    except:
        TOKEN = None
        print("   ⚠️ Токен не найден, результаты не сохранятся в GitHub")
        print("     (Добавьте GITHUB_TOKEN в Secrets Colab для сохранения)")
else:
    print("   ✅ Токен найден в переменных окружения")

# Сохраняем токен в переменную окружения для дочерних процессов
if TOKEN:
    os.environ['GITHUB_TOKEN'] = TOKEN

# ========== 2. КЛОНИРОВАНИЕ ==========
print("\n📥 Клонирование репозитория...")
repo_path = Path("/content/focus-vae-experiment")

if not repo_path.exists():
    if TOKEN:
        # Используем токен для клонирования
        clone_url = f"https://{TOKEN}@github.com/Alexeiyaganov/focus-vae-experiment.git"
        subprocess.run(f"git clone {clone_url}", shell=True, check=True)
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
    'models': ['vae', 'iwae', 'vamp', 'focus_vae']  
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

# ========== 8. 3D ВИЗУАЛИЗАЦИЯ ЛАТЕНТНОГО ПРОСТРАНСТВА ==========
print("\n" + "=" * 60)
print("🎨 3D ВИЗУАЛИЗАЦИЯ ЛАТЕНТНОГО ПРОСТРАНСТВА")
print("=" * 60)

try:
    # Устанавливаем plotly и необходимые пакеты
    subprocess.run("pip install plotly kaleido -q", shell=True)

    # Импортируем все необходимое
    import numpy as np
    import torch
    from torch.utils.data import DataLoader  # ВАЖНО: добавили импорт!
    from torchvision import datasets, transforms
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA


    def get_latent_codes(model, data_loader, device, n_samples=500):
        """Получение латентных кодов для визуализации"""
        model.eval()
        all_mu = []
        all_labels = []

        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader):
                if i * data_loader.batch_size >= n_samples:
                    break
                data = data.to(device)
                mu, _ = model.encoder(data.view(-1, 784))
                all_mu.append(mu.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_mu), np.concatenate(all_labels)


    # Создаем тестовый загрузчик
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Получаем латентные коды для каждой модели
    latent_spaces = {}

    for model_name in ['vae', 'iwae', 'focus_vae']:
        print(f"\n📊 Получение латентного пространства для {model_name}...")

        # Создаем модель с теми же параметрами
        if model_name == 'vae':
            model = VAE(config['latent_dim']).to(device)
        elif model_name == 'iwae':
            model = IWAE(config['latent_dim']).to(device)
        elif model_name == 'focus_vae':
            model = FocusVAE(config['latent_dim']).to(device)

        mu, labels = get_latent_codes(model, test_loader, device)

        # Уменьшаем размерность до 3D с помощью PCA
        pca = PCA(n_components=3)
        mu_3d = pca.fit_transform(mu)

        latent_spaces[model_name] = {
            'coords': mu_3d,
            'labels': labels,
            'explained_variance': pca.explained_variance_ratio_.sum()
        }

    # Создаем 3D визуализацию
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[f'{name.upper()} (PCA 3D)' for name in ['VAE', 'IWAE', 'FocusVAE']]
    )

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    for idx, (name, data) in enumerate(latent_spaces.items(), 1):
        coords = data['coords']
        labels = data['labels']

        for digit in range(10):
            mask = labels == digit
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors[digit],
                        opacity=0.8
                    ),
                    name=f'Цифра {digit}',
                    legendgroup=f'digit_{digit}',
                    showlegend=idx == 1
                ),
                row=1, col=idx
            )

    fig.update_layout(
        title='3D визуализация латентного пространства (PCA)',
        height=600
    )

    # Сохраняем как HTML для интерактивности
    plots_dir = Path('experiments/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / 'latent_space_3d.html'
    fig.write_html(str(html_path))

    # Сохраняем как PNG для GitHub
    png_path = plots_dir / 'latent_space_3d.png'
    fig.write_image(str(png_path))

    # Показываем в Colab
    fig.show()

    print(f"   ✅ 3D визуализация создана: {html_path}")

    # Добавляем в результаты для GitHub
    results['plots']['latent_space_3d'] = str(html_path)
    results['plots']['latent_space_3d_png'] = str(png_path)

except Exception as e:
    print(f"   ⚠️ Ошибка создания 3D визуализации: {e}")
    import traceback

    traceback.print_exc()



# ========== 9. СОХРАНЕНИЕ В GITHUB ==========
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

# ========== 10. ВЫВОД РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 60)
print("📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
print("=" * 60)

print("\n   " + "-" * 50)
print("   {:<12} | {:>12} | {:>12}".format("Модель", "Train Loss", "Test Loss"))
print("   " + "-" * 50)

# Находим лучшую модель
best_model_name = None
best_loss = float('inf')

for model_name in ['vae', 'iwae', 'vamp', 'focus_vae']:
    model_results = results['models'].get(model_name, {})
    train_loss = model_results.get('final_train_loss', 0)
    test_loss = model_results.get('test_loss', 0)

    if test_loss > 0 and test_loss < best_loss:
        best_loss = test_loss
        best_model_name = model_name

    winner = " 🏆" if model_name == best_model_name else ""
    print(f"   {model_name.upper():<12} | {train_loss:>12.2f} | {test_loss:>12.2f}{winner}")

print("   " + "-" * 50)

if best_model_name:
    print(f"\n🏆 Лучшая модель: {best_model_name.upper()} (Test Loss: {best_loss:.2f})")

print("\n" + "=" * 60)
print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("=" * 60)
print(f"\n📁 Результаты сохранены локально в: {repo_path}/experiments/")
if TOKEN:
    print(f"📤 Токен: {TOKEN[:4]}...{TOKEN[-4:]}")
    print("   Результаты отправлены в GitHub")
else:
    print("📤 Результаты не отправлены в GitHub (токен отсутствует)")
print("=" * 60)