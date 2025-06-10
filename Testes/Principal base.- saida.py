from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Carrega o modelo YOLO
modelo = YOLO('yolo12n.pt')

# Inicializa o dicionário para rastrear objetos
track_history = defaultdict(lambda: [])
# Contador de pessoas
contador_pessoas = 0
# Define uma linha virtual (ajuste as coordenadas conforme necessário)
linha_virtual = 300  # ajuste esta altura conforme seu vídeo

# Captura o vídeo
video = cv2.VideoCapture('exemplo3.mp4')

# Obtém as propriedades do vídeo original
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Configura o objeto VideoWriter para salvar o vídeo
output = cv2.VideoWriter('video_processado.mp4', 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Realiza a detecção com o YOLO
    resultados = modelo.track(frame, persist=True)

    if resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes.xyxy.cpu()  # Usando xyxy ao invés de xywh
        track_ids = resultados[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box  # Coordenadas já no formato correto
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Desenha a caixa delimitadora original do YOLO
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adiciona o ID do objeto
            cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            track = track_history[track_id]
            track.append((float(y1)))  # Usando y1 para tracking
            
            if len(track) > 2:
                # Verifica se o objeto atravessou a linha de cima para baixo
                if track[-2] < linha_virtual and track[-1] >= linha_virtual:
                    contador_pessoas += 1
                
                # Mantém apenas as últimas 30 posições
                track_history[track_id] = track[-30:]

    # Desenha a linha virtual
    cv2.line(frame, (0, linha_virtual), (frame.shape[1], linha_virtual), (0, 255, 0), 2)
    
    # Mostra o contador na tela
    cv2.putText(frame, f'Pessoas: {contador_pessoas}', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Salva o frame no vídeo de saída
    output.write(frame)

    # Mostra o frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
video.release()
output.release()
cv2.destroyAllWindows()

