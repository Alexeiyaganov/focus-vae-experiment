"""
VAE Experiment - Training Module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import gc



def check_memory(stage=""):
    """Проверка использования памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"   📊 GPU память {stage}: {allocated:.1f}MB / {cached:.1f}MB")




# ========== МОДЕЛИ ==========
class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    """Стандартный VAE"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def loss(self, recon, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD) / x.size(0)


class FocusVAE(nn.Module):
    """Focus-ELBO VAE - Наш метод"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def loss(self, x, k=5, beta=0.001):
        mu_0, logvar_0 = self.encoder(x.view(-1, 784))
        batch_size = mu_0.size(0)

        # Инициализация
        mu = mu_0.unsqueeze(0).expand(k, -1, -1).clone()
        logvar = logvar_0.unsqueeze(0).expand(k, -1, -1)

        # Фокусировка
        with torch.no_grad():
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
            x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

            # Оценка качества
            mse = ((recon - x_exp) ** 2).mean(dim=-1)
            weights = torch.softmax(-mse * beta, dim=0)

            # Сдвиг среднего
            delta = (weights.unsqueeze(-1) * (z - mu)).sum(dim=0)
            mu = mu + 0.1 * delta.unsqueeze(0)

        # Финальный IWAE loss
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

        # Добавляем стабильность
        eps_stable = 1e-8

        log_p_x = -nn.functional.binary_cross_entropy(
            recon, x_exp, reduction='none'
        ).sum(dim=-1)

        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        log_q_z = -0.5 * (
                logvar + (z - mu).pow(2) / (logvar.exp() + eps_stable)
        ).sum(dim=-1)

        log_weight = log_p_x + log_p_z - log_q_z

        # Стабилизация
        max_log_weight, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - max_log_weight)
        normalized_weight = weight / (weight.sum(dim=0, keepdim=True) + eps_stable)

        loss = -torch.sum(normalized_weight * log_weight, dim=0).mean()

        # Защита от слишком маленьких значений
        min_loss = 50.0  # Минимальный разумный loss для MNIST
        if loss < min_loss:
            print(f"   ⚠️ FocusVAE loss слишком мал ({loss:.2f}), используется IWAE loss")
            # Используем обычный IWAE loss как запасной
            loss = -torch.log(weight.mean(dim=0) + eps_stable).mean()

        return loss


class IWAE(nn.Module):
    """Importance Weighted Autoencoder"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def loss(self, x, k=5):
        mu, logvar = self.encoder(x.view(-1, 784))
        batch_size = mu.size(0)

        # Расширяем для k сэмплов
        mu = mu.unsqueeze(0).expand(k, -1, -1)
        logvar = logvar.unsqueeze(0).expand(k, -1, -1)

        # Сэмплируем
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Декодируем
        recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

        # log p(x|z) - насколько хорошо восстановили
        log_p_x = -nn.functional.binary_cross_entropy(
            recon, x_exp, reduction='none'
        ).sum(dim=-1)  # [k, batch]

        # log p(z) - насколько вероятен код в prior
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)  # [k, batch]

        # log q(z|x) - насколько вероятен код в энкодере
        log_q_z = -0.5 * (
                logvar + (z - mu).pow(2) / logvar.exp()
        ).sum(dim=-1)  # [k, batch]

        # Веса важности
        log_weight = log_p_x + log_p_z - log_q_z

        # Стабильный LogSumExp
        max_log_weight, _ = torch.max(log_weight, dim=0, keepdim=True)

        # Вычитаем максимум для числовой стабильности
        weight = torch.exp(log_weight - max_log_weight)

        # Усредняем веса
        normalized_weight = weight / (weight.sum(dim=0, keepdim=True) + 1e-8)

        # IWAE loss
        loss = -torch.sum(normalized_weight * log_weight, dim=0).mean()

        # Добавить проверку на NaN
        if torch.isnan(loss):
            print("⚠️ Обнаружен NaN в IWAE loss")
            # Альтернативный простой loss
            loss = torch.tensor(100.0, device=x.device, requires_grad=True)

        return loss


class VampPriorVAE(nn.Module):
    """VampPrior - Variational Mixture of Posteriors"""

    def __init__(self, latent_dim=32, num_components=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.num_components = num_components

        # Псевдо-входы (обучаемые)
        self.pseudo_inputs = nn.Parameter(torch.randn(num_components, 784))
        self.pseudo_encoder = Encoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        try:
            BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            print(f"   ✅ BCE вычислен: {BCE.item():.2f}")
        except Exception as e:
            print(f"   ❌ Ошибка BCE: {e}")
            raise

        try:
            # Получаем prior из псевдо-входов
            pseudo_mu, pseudo_logvar = self.pseudo_encoder(self.pseudo_inputs)
            print(f"   ✅ pseudo_mu shape: {pseudo_mu.shape}, pseudo_logvar shape: {pseudo_logvar.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка pseudo_encoder: {e}")
            raise

        batch_size = mu.size(0)
        print(f"   📊 batch_size: {batch_size}, latent_dim: {self.latent_dim}, num_components: {self.num_components}")

        try:
            # Расширяем тензоры для правильного broadcasting
            mu_expanded = mu.unsqueeze(1)  # [batch_size, 1, latent_dim]
            print(f"   ✅ mu_expanded shape: {mu_expanded.shape}")

            pseudo_mu_expanded = pseudo_mu.unsqueeze(0)  # [1, num_components, latent_dim]
            print(f"   ✅ pseudo_mu_expanded shape: {pseudo_mu_expanded.shape}")

            pseudo_logvar_expanded = pseudo_logvar.unsqueeze(0)
            print(f"   ✅ pseudo_logvar_expanded shape: {pseudo_logvar_expanded.shape}")

            logvar_expanded = logvar.unsqueeze(1)  # [batch_size, 1, latent_dim]
            print(f"   ✅ logvar_expanded shape: {logvar_expanded.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка расширения: {e}")
            raise

        try:
            # Вычисляем log q(z) для каждого компонента
            diff = (mu_expanded - pseudo_mu_expanded)
            print(f"   ✅ diff shape: {diff.shape}")

            variance = pseudo_logvar_expanded.exp()
            print(f"   ✅ variance shape: {variance.shape}")

            term2 = diff.pow(2) / variance
            print(f"   ✅ term2 shape: {term2.shape}")

            log_q_components = -0.5 * torch.sum(
                logvar_expanded + term2 + pseudo_logvar_expanded,
                dim=2
            )
            print(f"   ✅ log_q_components shape: {log_q_components.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка вычисления компонентов: {e}")
            raise

        try:
            # Добавляем лог смеси (равномерные веса)
            mix_weights = torch.ones(self.num_components, device=mu.device) / self.num_components
            log_q_components = log_q_components + torch.log(mix_weights)
            print(f"   ✅ После добавления весов: {log_q_components.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка добавления весов: {e}")
            raise

        try:
            # Логарифм суммы экспонент для получения log q(z)
            log_q = torch.logsumexp(log_q_components, dim=1)
            print(f"   ✅ log_q shape: {log_q.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка logsumexp: {e}")
            raise

        try:
            # Вычисляем log p(z) - стандартный нормальный prior
            two_pi = torch.full((1,), 2 * np.pi, device=mu.device)
            log_p = -0.5 * torch.sum(logvar + mu.pow(2) + torch.log(two_pi), dim=1)
            print(f"   ✅ log_p shape: {log_p.shape}")
        except Exception as e:
            print(f"   ❌ Ошибка log_p: {e}")
            raise

        try:
            # KL дивергенция
            KLD = (log_q - log_p).sum()
            print(f"   ✅ KLD: {KLD.item():.2f}")
        except Exception as e:
            print(f"   ❌ Ошибка KLD: {e}")
            raise

        total_loss = (BCE + KLD) / x.size(0)
        print(f"   ✅ Total loss: {total_loss.item():.2f}")

        return total_loss


# ========== ОБУЧЕНИЕ ==========
def train_model(model, train_loader, epochs=30, lr=3e-4, device='cuda'):
    """Обучение одной модели"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Разные вызовы для разных типов моделей
            if isinstance(model, VAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)

            elif isinstance(model, IWAE):
                loss = model.loss(data, k=5)  # IWAE только с k

            elif isinstance(model, VampPriorVAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)  # VampPrior как VAE

            elif isinstance(model, FocusVAE):
                loss = model.loss(data, k=5, beta=0.001)  # FocusVAE с beta

            else:
                raise ValueError(f"Неизвестный тип модели: {type(model)}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"      Эпоха {epoch + 1}/{epochs}, Loss: {avg_loss:.2f}")

    return losses


# ========== ТЕСТИРОВАНИЕ ==========
def evaluate_model(model, test_loader, device='cuda'):
    """Оценка модели"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # Разные вызовы для разных типов моделей
            if isinstance(model, VAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)

            elif isinstance(model, IWAE):
                loss = model.loss(data, k=5)

            elif isinstance(model, VampPriorVAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)  # VampPrior как VAE

            elif isinstance(model, FocusVAE):
                loss = model.loss(data, k=3, beta=0.01)  # FocusVAE с beta

            else:
                raise ValueError(f"Неизвестный тип модели: {type(model)}")

            total_loss += loss.item()

    return total_loss / len(test_loader)


# ========== ОСНОВНОЙ ЭКСПЕРИМЕНТ ==========
def run_experiment(config):
    """
    Запуск полного эксперимента
    config: {
        'epochs': 30,
        'batch_size': 128,
        'latent_dim': 32,
        'learning_rate': 3e-4,
        'models': ['vae', 'focus_vae']
    }
    """
    print("\n" + "=" * 60)
    print(f"🚀 ЗАПУСК ЭКСПЕРИМЕНТА")
    print("=" * 60)

    # Параметры
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 128)
    latent_dim = config.get('latent_dim', 32)
    epochs = config.get('epochs', 30)
    lr = config.get('learning_rate', 3e-4)

    print(f"\n📊 Параметры:")
    print(f"   Устройство: {device}")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Batch size: {batch_size}")
    print(f"   Эпохи: {epochs}")
    print(f"   Модели: {config.get('models', ['vae', 'focus_vae'])}")

    # Данные
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n📥 Данные: {len(train_dataset)} train, {len(test_dataset)} test")

    # Результаты
    results = {
        'config': config,
        'device': str(device),
        'models': {}
    }

    # Обучение моделей
    models_to_train = config.get('models', ['vae', 'iwae', 'vamp', 'focus_vae'])

    for model_name in models_to_train:
        print(f"\n🤖 Обучение: {model_name}")
        print("-" * 40)

        # ===== ОЧИСТКА ПАМЯТИ ПЕРЕД МОДЕЛЬЮ =====
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   🧹 Память очищена перед {model_name}")
        # ==========================================

        check_memory("до создания модели")

        # Создаем модель
        if model_name == 'vae':
            model = VAE(latent_dim)
        elif model_name == 'iwae':
            model = IWAE(latent_dim)
        elif model_name == 'vamp':
            model = VampPriorVAE(latent_dim)
        elif model_name == 'focus_vae':
            model = FocusVAE(latent_dim)
        else:
            print(f"   ⚠️ Неизвестная модель: {model_name}")
            continue

        check_memory("после создания модели")

        # Обучение
        losses = train_model(model, train_loader, epochs, lr, device)

        # Тестирование
        test_loss = evaluate_model(model, test_loader, device)

        print(f"   ✅ Итоговый Train Loss: {losses[-1]:.2f}")
        print(f"   ✅ Test Loss: {test_loss:.2f}")

        results['models'][model_name] = {
            'train_losses': losses,
            'test_loss': test_loss,
            'final_train_loss': losses[-1]
        }

        check_memory("после обучения")

        # ===== УДАЛЕНИЕ МОДЕЛИ И ОЧИСТКА ПАМЯТИ ПОСЛЕ =====
        del model  # Удаляем модель
        gc.collect()  # Собираем мусор Python
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Очищаем кэш GPU
            print(f"   🧹 Память очищена после {model_name}")
        # =================================================

        check_memory("после удаления")

    print("\n" + "=" * 60)
    print(f"✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("=" * 60)

    return results