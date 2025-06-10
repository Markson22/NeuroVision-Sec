from ultralytics import YOLO
from Configuração_treino import CONFIG, TRAIN_PARAMS
import yaml
from pathlib import Path

def treinar_modelo():
    """Função principal de treinamento com CPU"""
    try:
        # Cria arquivo de configuração YAML
        yaml_path = Path('dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(CONFIG, f)
        
        # Inicializa modelo YOLO
        model = YOLO('yolov8n.pt')  # Usando modelo menor para CPU
        
        # Configura parâmetros otimizados para CPU
        train_args = {
            'data': str(yaml_path),
            'epochs': TRAIN_PARAMS['epochs'],
            'imgsz': TRAIN_PARAMS['img_size'],
            'batch': TRAIN_PARAMS['batch_size'],
            'device': 'cpu',
            'workers': TRAIN_PARAMS['workers'],
            'name': 'weapon_detector_cpu',
            'patience': 20,
            'save': True
        }
        
        print("Iniciando treinamento na CPU...")
        print("Aviso: O treinamento será mais lento sem GPU!")
        results = model.train(**train_args)
        
        return True
        
    except Exception as e:
        print(f"Erro durante treinamento: {str(e)}")
        return False

if __name__ == "__main__":
    treinar_modelo()