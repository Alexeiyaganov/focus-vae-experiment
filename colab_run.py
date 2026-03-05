"""
ОДНА КОМАНДА для запуска в Colab
Скачивает, обучает, сохраняет в GitHub
"""

import os
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
subprocess.run("pip install plotly kaleido -q", shell=True)

# Импортируем все необходимое
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from train import VAE, IWAE, FocusVAE


# Для красивых графиков
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2


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
    'models': ['vae', 'iwae','focus_vae']   #'vamp'
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

    # Настройка стиля для красивых графиков
    plt.style.use('seaborn-v0_8-darkgrid')

    # График сходимости
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Цвета для разных моделей
    colors = {
        'vae': '#1f77b4',  # синий
        'iwae': '#ff7f0e',  # оранжевый
        'vamp': '#2ca02c',  # зеленый
        'focus_vae': '#d62728'  # красный
    }

    # Левый график: обычный
    ax = axes[0]
    for model_name, model_results in results['models'].items():
        losses = model_results.get('train_losses', [])
        if losses:
            ax.plot(range(1, len(losses) + 1), losses,
                    label=model_name.upper(),
                    color=colors.get(model_name, 'gray'),
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    markevery=max(1, len(losses) // 5))

    ax.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss (ELBO)', fontsize=13, fontweight='bold')
    ax.set_title('Сравнение скорости сходимости', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)  # Начинаем с 0

    # Правый график: логарифмический масштаб
    ax = axes[1]
    for model_name, model_results in results['models'].items():
        losses = model_results.get('train_losses', [])
        if losses:
            ax.semilogy(range(1, len(losses) + 1), losses,
                        label=model_name.upper(),
                        color=colors.get(model_name, 'gray'),
                        linewidth=2.5,
                        marker='s',
                        markersize=4,
                        markevery=max(1, len(losses) // 5))

    ax.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss (ELBO) - лог. шкала', fontsize=13, fontweight='bold')
    ax.set_title('Сходимость в логарифмическом масштабе', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    plt.tight_layout()

    # Сохраняем локально
    plots_dir = Path('experiments/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / 'convergence_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
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

    # Определяем device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   📊 Используется устройство: {device}")


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

        # Для лучшей визуализации используем t-SNE вместо PCA
        print(f"      Выполняется t-SNE (может занять минуту)...")
        tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
        mu_3d = tsne.fit_transform(mu[:500])  # Меньше точек для скорости

        latent_spaces[model_name] = {
            'coords': mu_3d,
            'labels': labels[:500],
        }

    # Создаем 3D визуализацию
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[f'{name.upper()} (3D вид)' for name in ['VAE', 'IWAE', 'FocusVAE']] * 2
    )

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # Верхний ряд - обычный вид
    for idx, (name, data) in enumerate(latent_spaces.items(), 1):
        coords = data['coords']
        labels = data['labels']

        for digit in range(10):
            mask = labels == digit
            if mask.sum() > 5:  # Минимум 5 точек для показа
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        z=coords[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=colors[digit],
                            opacity=0.7
                        ),
                        name=f'Цифра {digit}',
                        legendgroup=f'digit_{digit}',
                        showlegend=idx == 1
                    ),
                    row=1, col=idx
                )

    # Нижний ряд - вид с выделенными кластерами
    for idx, (name, data) in enumerate(latent_spaces.items(), 1):
        coords = data['coords']
        labels = data['labels']

        # Добавляем эллипсоиды для каждого кластера
        for digit in range(10):
            mask = labels == digit
            if mask.sum() > 10:
                cluster_points = coords[mask]
                center = cluster_points.mean(axis=0)

                # Добавляем прозрачную сферу для показа плотности
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)

                # Размер сферы зависит от разброса точек
                radius = np.std(cluster_points, axis=0).mean() * 2

                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v)).flatten()
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v)).flatten()
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()

                fig.add_trace(
                    go.Scatter3d(
                        x=x_sphere,
                        y=y_sphere,
                        z=z_sphere,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=colors[digit],
                            opacity=0.1
                        ),
                        name=f'Кластер {digit}',
                        legendgroup=f'cluster_{digit}',
                        showlegend=False
                    ),
                    row=2, col=idx
                )

                # Добавляем центр кластера
                fig.add_trace(
                    go.Scatter3d(
                        x=[center[0]],
                        y=[center[1]],
                        z=[center[2]],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[digit],
                            symbol='diamond'
                        ),
                        showlegend=False
                    ),
                    row=2, col=idx
                )

    fig.update_layout(
        title='Сравнение латентных пространств: VAE vs IWAE vs FocusVAE',
        height=800,
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        )
    )

    # Сохраняем
    plots_dir = Path('experiments/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / 'latent_space_improved.html'
    fig.write_html(str(html_path))

    # Показываем в Colab
    fig.show()

    print(f"   ✅ Улучшенная 3D визуализация создана: {html_path}")

    # Добавляем в результаты
    results['plots']['latent_space_improved'] = str(html_path)

except Exception as e:
    print(f"   ⚠️ Ошибка создания 3D визуализации: {e}")
    import traceback

    traceback.print_exc()

# ========== 9. ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ФОКУСИРОВКИ FOCUSVAE ==========
print("\n" + "=" * 60)
print("🎯 ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ФОКУСИРОВКИ FOCUSVAE")
print("=" * 60)


try:
    if 'focus_vae' in results['models']:
        # Создаем модель FocusVAE
        model_focus = FocusVAE(config['latent_dim']).to(device)

        # Берем несколько изображений для демонстрации
        test_loader_demo = DataLoader(test_dataset, batch_size=4, shuffle=True)
        samples, labels = next(iter(test_loader_demo))
        samples = samples.to(device)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        with torch.no_grad():
            # Получаем реконструкции
            recon_focus, _, _ = model_focus(samples.view(-1, 784))

            # Для сравнения - VAE
            model_vae = VAE(config['latent_dim']).to(device)
            recon_vae, _, _ = model_vae(samples.view(-1, 784))

        for i in range(4):
            # Оригинал
            axes[i, 0].imshow(samples[i].cpu().squeeze(), cmap='gray')
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Оригинал', fontsize=10, fontweight='bold')

            # VAE
            axes[i, 1].imshow(recon_vae[i].cpu().view(28, 28), cmap='gray')
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('VAE', fontsize=10, fontweight='bold')

            # FocusVAE
            axes[i, 2].imshow(recon_focus[i].cpu().view(28, 28), cmap='gray')
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('FocusVAE', fontsize=10, fontweight='bold')

            # Разница (FocusVAE - оригинал)
            diff_focus = torch.abs(recon_focus[i].cpu().view(28, 28) - samples[i].cpu().squeeze())
            axes[i, 3].imshow(diff_focus, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 3].axis('off')
            if i == 0:
                axes[i, 3].set_title('Ошибка FocusVAE', fontsize=10, fontweight='bold')

        plt.suptitle('Сравнение реконструкций: VAE vs FocusVAE', fontsize=14, fontweight='bold')
        plt.tight_layout()

        compare_plot_path = plots_dir / 'reconstruction_comparison.png'
        plt.savefig(compare_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"   ✅ Сравнение реконструкций создано: {compare_plot_path}")
        results['plots']['reconstruction_comparison'] = str(compare_plot_path)

except Exception as e:
    print(f"   ⚠️ Ошибка визуализации фокусировки: {e}")
    import traceback

    traceback.print_exc()



# ========== 10. СОХРАНЕНИЕ В GITHUB ==========
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



# ========== 11. УЛУЧШЕННЫЙ ВЫВОД РЕЗУЛЬТАТОВ ==========
print("\n" + "=" * 70)
print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
print("=" * 70)

# Собираем данные для таблицы
model_data = []
for model_name in ['vae', 'iwae', 'vamp', 'focus_vae']:
    model_results = results['models'].get(model_name, {})
    if model_results:  # Только если модель обучалась
        train_loss = model_results.get('final_train_loss', 0)
        test_loss = model_results.get('test_loss', 0)
        model_data.append({
            'name': model_name.upper(),
            'train_loss': train_loss,
            'test_loss': test_loss
        })

# Находим лучшую модель
best_model = min(model_data, key=lambda x: x['test_loss']) if model_data else None

# Выводим красивую таблицу
print("\n┌" + "─" * 68 + "┐")
print("│ {:<12} │ {:>15} │ {:>15} │ {:>15} │".format("Модель", "Train Loss", "Test Loss", "Улучшение"))
print("├" + "─" * 68 + "┤")

vae_loss = next((m['test_loss'] for m in model_data if m['name'] == 'VAE'), None)

for model in sorted(model_data, key=lambda x: x['test_loss']):
    name = model['name']
    train = model['train_loss']
    test = model['test_loss']

    # Рассчитываем улучшение относительно VAE
    if vae_loss and name != 'VAE':
        improvement = ((vae_loss - test) / vae_loss) * 100
        imp_str = f"{improvement:+.2f}%"
    else:
        imp_str = "—"

    # Отмечаем лучшую модель
    if best_model and name == best_model['name']:
        name = f"★ {name} ★"
        winner_mark = " 🏆"
    else:
        winner_mark = ""

    print(f"│ {name:<12} │ {train:>15.2f} │ {test:>15.2f}{winner_mark} │ {imp_str:>15} │")

print("└" + "─" * 68 + "┘")

if best_model:
    print(f"\n🏆 АБСОЛЮТНЫЙ ПОБЕДИТЕЛЬ: {best_model['name']} с loss {best_model['test_loss']:.2f}")

    # Дополнительная статистика
    if vae_loss:
        improvement = ((vae_loss - best_model['test_loss']) / vae_loss) * 100
        print(f"📈 Улучшение относительно VAE: {improvement:.2f}%")

# Выводим информацию о времени выполнения
print("\n" + "=" * 70)
print("✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН")
print("=" * 70)
print(f"\n📁 Результаты сохранены локально в: {repo_path}/experiments/")
print(f"📊 Создано графиков: {len(results.get('plots', {}))}")
if TOKEN:
    print(f"📤 Результаты отправлены в GitHub (токен: {TOKEN[:4]}...{TOKEN[-4:]})")
else:
    print("📤 Результаты не отправлены в GitHub (токен отсутствует)")
print("=" * 70)