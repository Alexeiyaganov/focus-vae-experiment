#!/usr/bin/env python3
"""
ОДНА КОМАНДА для полного запуска в Colab
"""

import os

# Установка и настройка
os.system("wget -q -O /tmp/setup.py https://raw.githubusercontent.com/Alexeiyaganov/focus-vae-experiment/main/scripts/colab_setup.py")
os.system("python /tmp/setup.py")

print("\n" + "=" * 60)
print("🎯 ГОТОВО! ТЕПЕРЬ ВЫПОЛНИТЕ:")
print("=" * 60)
print("""
from scripts.create_job import create_quick_test
from scripts.worker import start_worker

# 1. Создать задание
job_id = create_quick_test()

# 2. Запустить обработку
start_worker(check_interval=30, max_jobs=5)
""")