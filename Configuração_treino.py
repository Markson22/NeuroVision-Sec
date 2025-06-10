from pathlib import Path

# Configuração do dataset
CONFIG = {
    'path': str(Path('Dataset/archive/weapon_detection')),  # caminho do dataset
    'train': 'train/images',  # pasta de imagens de treino
    'val': 'val/images',      # pasta de imagens de validação
    'test': 'test/images',    # pasta de imagens de teste
    'names': {
        0: 'weapon'           # classes para detecção
    }
}

# Configurações de treinamento
TRAIN_PARAMS = {
    'epochs': 100,
    'batch_size': 4,  # Reduzido para não sobrecarregar a CPU
    'img_size': 640,
    'device': 'cpu',  # Forçando uso da CPU
    'workers': 4      # Reduzido para melhor performance na CPU
}