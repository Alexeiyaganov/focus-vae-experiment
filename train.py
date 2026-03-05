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
    """Focus-ELBO VAE - ПРОСТАЯ РАБОЧАЯ ВЕРСИЯ"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

        # Минимальный refiner
        self.refiner = nn.Linear(latent_dim * 2, latent_dim)

        self.beta = 0.01  # Фиксированный маленький шаг

    def forward(self, x):
        """Forward pass для визуализации"""
        mu, logvar = self.encoder(x.view(-1, 784))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def iwae_loss(self, mu, logvar, x, k=5):
        """Обычный IWAE loss"""
        batch_size = mu.size(0)
        x_flat = x.view(-1, 784)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn(k, batch_size, self.latent_dim, device=x.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)

        recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x_flat.unsqueeze(0).expand(k, -1, -1)

        log_p_x = -nn.functional.binary_cross_entropy(
            recon, x_exp, reduction='none'
        ).sum(-1)

        log_p_z = -0.5 * (z ** 2).sum(-1)
        log_q_z = -0.5 * (logvar + (z - mu).pow(2) / (logvar.exp() + 1e-8)).sum(-1)

        log_weight = log_p_x + log_p_z - log_q_z
        max_log_weight, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - max_log_weight)

        return -torch.log(weight.mean(dim=0) + 1e-8).mean()

    def loss(self, x, k=5, focus_steps=1, training=True):
        x_flat = x.view(-1, 784)
        batch_size = x_flat.size(0)

        mu_0, logvar_0 = self.encoder(x_flat)

        # Отладка
        print(f"      mu_0 mean: {mu_0.mean().item():.4f}, logvar_0 mean: {logvar_0.mean().item():.4f}")

        if not training:
            return self.iwae_loss(mu_0, logvar_0, x, k=k)

        # Простая фокусировка - всего один шаг
        mu_curr = mu_0.clone().detach().requires_grad_(True)

        std = torch.exp(0.5 * logvar_0)
        eps = torch.randn(k, batch_size, self.latent_dim, device=x.device)
        z = mu_curr.unsqueeze(0) + eps * std.unsqueeze(0)

        recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x_flat.unsqueeze(0).expand(k, -1, -1)

        log_p_x = -nn.functional.binary_cross_entropy(
            recon, x_exp, reduction='none'
        ).sum(-1)
        log_p_z = -0.5 * (z ** 2).sum(-1)
        log_q_z = -0.5 * (logvar_0 + (z - mu_curr).pow(2) / (logvar_0.exp() + 1e-8)).sum(-1)

        advantage = log_p_x + log_p_z - log_q_z
        grad_mu = torch.autograd.grad(advantage.mean(), mu_curr, retain_graph=False)[0]

        # Нормализуем градиент
        grad_norm = torch.norm(grad_mu, dim=-1, keepdim=True)
        grad_normalized = grad_mu / (grad_norm + 1e-8)

        # Простое уточнение
        refinement = self.refiner(torch.cat([mu_curr, grad_normalized], dim=-1))
        mu_new = mu_curr + self.beta * refinement

        # Не уходим далеко
        mu_final = 0.9 * mu_new + 0.1 * mu_0

        loss = self.iwae_loss(mu_final, logvar_0, x, k=k)

        # Небольшая регуляризация
        reg = 0.01 * torch.sum((mu_final - mu_0).pow(2)) / batch_size

        return loss + reg


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
    """VampPrior - Variational Mixture of Posteriors (ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ)"""

    def __init__(self, latent_dim=32, num_components=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.num_components = num_components

        # Псевдо-входы (обучаемые)
        self.pseudo_inputs = nn.Parameter(torch.randn(num_components, 784))
        self.pseudo_encoder = Encoder(latent_dim)

        # Константы (вычисляются один раз при создании)
        self.register_buffer('pi_constant', torch.tensor(np.pi))
        self.register_buffer('two_pi_constant', torch.tensor(2 * np.pi))
        self.register_buffer('log_2pi_constant', torch.tensor(np.log(2 * np.pi)))

        print(f"   ✅ VampPrior инициализирован с {num_components} компонентами")

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # 1. BCE loss (реконструкция)
        BCE = nn.functional.binary_cross_entropy(
            recon_x, x.view(-1, 784), reduction='sum'
        )

        batch_size = mu.size(0)

        # 2. Получаем prior из псевдо-входов
        pseudo_mu, pseudo_logvar = self.pseudo_encoder(self.pseudo_inputs)

        # 3. Вычисляем log q(z) - апостериорное
        # Расширяем для broadcasting
        mu_exp = mu.unsqueeze(1)  # [B, 1, D]
        pseudo_mu_exp = pseudo_mu.unsqueeze(0)  # [1, C, D]
        pseudo_logvar_exp = pseudo_logvar.unsqueeze(0)  # [1, C, D]

        # Вычисляем расстояние Махаланобиса для каждого компонента
        diff = mu_exp - pseudo_mu_exp  # [B, C, D]

        # Используем заранее вычисленную константу
        log_2pi = self.log_2pi_constant.to(mu.device)

        # log q(z) для каждого компонента: [B, C]
        log_q_components = -0.5 * torch.sum(
            log_2pi +
            pseudo_logvar_exp +
            diff.pow(2) / (pseudo_logvar_exp.exp() + 1e-8),
            dim=2
        )

        # Равномерные веса смеси
        log_weights = torch.log(torch.ones(self.num_components, device=mu.device) / self.num_components)
        log_q_components = log_q_components + log_weights

        # log q(z) = logsumexp over components
        log_q = torch.logsumexp(log_q_components, dim=1)

        # 4. Вычисляем log p(z) - стандартный нормальный prior
        log_p = -0.5 * torch.sum(
            logvar + mu.pow(2) + log_2pi,
            dim=1
        )

        # 5. KL дивергенция
        KLD = (log_q - log_p).sum()

        # 6. Финальный loss
        total_loss = (BCE + KLD) / batch_size

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

            if isinstance(model, VAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)
            elif isinstance(model, IWAE):
                loss = model.loss(data, k=5)
            elif isinstance(model, VampPriorVAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)
            elif isinstance(model, FocusVAE):
                loss = model.loss(data, k=8, focus_steps=1, training=True)
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

def evaluate_model(model, test_loader, device='cuda'):
    """Оценка модели"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            if isinstance(model, VAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)
            elif isinstance(model, IWAE):
                loss = model.loss(data, k=5)
            elif isinstance(model, VampPriorVAE):
                recon, mu, logvar = model(data.view(-1, 784))
                loss = model.loss(recon, data, mu, logvar)
            elif isinstance(model, FocusVAE):
                loss = model.loss(data, k=8, focus_steps=2, training=False)  # training=False
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