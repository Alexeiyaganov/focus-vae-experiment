# Focus-ELBO VAE Experiment System

🚀 Система для проведения экспериментов с VAE моделями на Google Colab с автоматическим сохранением в GitHub.

## 📋 Быстрый старт

### 1. Настройка GitHub
1. Создайте репозиторий: `focus-vae-experiment`
2. Загрузите все файлы из этой папки

### 2. Настройка Google Colab
1. Откройте [Google Colab](https://colab.research.google.com/)
2. Добавьте GitHub Token:
   - Левая панель → 🔑 Secrets (NOTA BENE)
   - `+ Add new secret`
   - Name: `GITHUB_TOKEN`
   - Value: [ваш токен GitHub](https://github.com/settings/tokens)

### 3. Запуск в Colab
```python
# Вставьте в ячейку Colab и выполните:
!wget -q -O /tmp/setup.py https://raw.githubusercontent.com/Alexeiyaganov/focus-vae-experiment/main/scripts/colab_setup.py
%run /tmp/setup.py