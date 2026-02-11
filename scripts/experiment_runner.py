"""
Основной движок экспериментов с VAE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import time

# =================== АРХИТЕКТУРЫ VAE ===================

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


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
    """Базовый VAE"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(
            recon_x, x.view(-1, 784), reduction='sum'
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD) / x.size(0)


class IWAE(VAE):
    """IWAE - Importance Weighted Autoencoder"""
    def loss(self, x, k=5):
        mu, logvar = self.encoder(x.view(-1, 784))
        batch_size = mu.size(0)

        mu = mu.unsqueeze(0).expand(k, -1, -1)
        logvar = logvar.unsqueeze(0).expand(k, -1, -1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

        log_p_x_given_z = -nn.functional.binary_cross_entropy(
            recon, x_exp, reduction='none'
        ).sum(dim=-1)

        log_p_z = -0.5 * (z ** 2).sum(dim=-1)
        log_q_z_given_x = -0.5 * (
            logvar + (z - mu) ** 2 / torch.exp(logvar)
        ).sum(dim=-1)

        log_weight = log_p_x_given_z + log_p_z - log_q_z_given_x

        max_log_weight, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - max_log_weight)
        loss = -torch.log(weight.mean(dim=0) + 1e-8) - max_log_weight.squeeze(0)

        return loss.mean()


class FocusELBO(VAE):
    """Focus-ELBO - наш метод с адаптивной фокусировкой"""
    def loss(self, x, k=5, focus_steps=1, beta=0.01):
        mu_0, logvar_0 = self.encoder(x.view(-1, 784))
        batch_size = mu_0.size(0)

        k = min(k, 3)
        mu = mu_0.unsqueeze(0).expand(k, -1, -1).clone()
        logvar = logvar_0.unsqueeze(0).expand(k, -1, -1)

        if focus_steps > 0:
            with torch.no_grad():
                for _ in range(focus_steps):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std

                    recon = self.decoder(z.view(-1, self.latent_dim)).view(k, batch_size, -1)
                    x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

                    # Вычисляем advantage
                    mse = ((recon - x_exp) ** 2).mean(dim=-1)
                    advantage = -mse

                    advantage_norm = (advantage - advantage.mean(dim=0, keepdim=True)) / \
                                    (advantage.std(dim=0, keepdim=True) + 1e-8)

                    weights = torch.softmax(beta * advantage_norm, dim=0)
                    delta = (weights.unsqueeze(-1) * (z - mu)).sum(dim=0)

                    mu = mu + 0.05 * delta.unsqueeze(0)
                    mu = 0.9 * mu + 0.1 * mu_0.unsqueeze(0).expand(k, -1, -1)

        # Финальный IWAE loss
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_final = mu + eps * std

        recon_final = self.decoder(z_final.view(-1, self.latent_dim)).view(k, batch_size, -1)
        x_exp = x.view(-1, 784).unsqueeze(0).expand(k, -1, -1)

        log_p_x_given_z = -nn.functional.binary_cross_entropy(
            recon_final, x_exp, reduction='none'
        ).sum(dim=-1)

        log_p_z = -0.5 * (z_final ** 2).sum(dim=-1)
        log_q_z_given_x = -0.5 * (
            logvar + (z_final - mu) ** 2 / torch.exp(logvar)
        ).sum(dim=-1)

        log_weight = log_p_x_given_z + log_p_z - log_q_z_given_x
        max_log_weight, _ = torch.max(log_weight, dim=0, keepdim=True)
        weight = torch.exp(log_weight - max_log_weight)

        return -torch.log(weight.mean(dim=0) + 1e-10) - max_log_weight.squeeze(0).mean()


# =================== ЗАПУСК ЭКСПЕРИМЕНТА ===================

def run_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Запуск эксперимента с заданной конфигурацией
    """
    print("\n" + "=" * 60)
    print(f"🧪 ЗАПУСК ЭКСПЕРИМЕНТА")
    print("=" * 60)

    # Параметры
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 64)
    latent_dim = config.get('latent_dim', 32)
    models_to_train = config.get('models', ['vae'])

    print(f"📊 Параметры:")
    print(f"   - Устройство: {device}")
    print(f"   - Эпохи: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Latent dim: {latent_dim}")
    print(f"   - Модели: {models_to_train}")

    # Загрузка данных
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = {
        'config': config,
        'device': str(device),
        'models': {},
        'timestamp': datetime.now().isoformat()
    }

    # Обучение каждой модели
    for model_name in models_to_train:
        print(f"\n🤖 Обучение модели: {model_name}")
        print("-" * 40)

        # Создаем модель
        if model_name == 'vae':
            model = VAE(latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        elif model_name == 'iwae':
            model = IWAE(latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        elif model_name == 'focus_elbo':
            model = FocusELBO(latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        else:
            print(f"⚠️ Неизвестная модель: {model_name}")
            continue

        # Обучение
        model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()

                if model_name == 'vae':
                    recon, mu, logvar = model(data.view(-1, 784))
                    loss = model.loss(recon, data, mu, logvar)
                elif model_name == 'iwae':
                    loss = model.loss(data, k=config.get('k_samples', 5))
                elif model_name == 'focus_elbo':
                    loss = model.loss(
                        data,
                        k=config.get('k_samples', 3),
                        focus_steps=config.get('focus_steps', 1),
                        beta=config.get('beta', 0.01)
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"   Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Оценка
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                if model_name == 'vae':
                    recon, mu, logvar = model(data.view(-1, 784))
                    loss = model.loss(recon, data, mu, logvar)
                else:
                    loss = model.loss(data, k=config.get('k_samples', 5))
                test_loss += loss.item()

        test_loss /= len(test_loader)

        # Сохраняем результаты
        results['models'][model_name] = {
            'final_train_loss': losses[-1],
            'test_loss': test_loss,
            'loss_history': losses,
            'config': {
                'latent_dim': latent_dim,
                'k_samples': config.get('k_samples', 5),
                'focus_steps': config.get('focus_steps', 1),
                'beta': config.get('beta', 0.01)
            }
        }

        # Сохраняем модель
        torch.save(model.state_dict(), output_dir / f"{model_name}.pth")

        print(f"   ✅ {model_name} - Test Loss: {test_loss:.4f}")

    # Сохраняем результаты
    with open(output_dir / "full_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("=" * 60)

    return results