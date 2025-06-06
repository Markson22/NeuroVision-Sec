from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Configurações
RTMP_URL = 'rtmp://10.0.88.6/live/cam1'
LINHA_VIRTUAL = 800

# Carrega o modelo YOLO
modelo = YOLO('yolo11n.pt')

# Inicializa contadores e histórico
track_history = defaultdict(lambda: [])
pessoas_entrando = 0
pessoas_saindo = 0

# Captura o vídeo da câmera RTMP
video = cv2.VideoCapture(RTMP_URL)

# Verifica se a conexão foi estabelecida
if not video.isOpened():
    print("Erro ao conectar com a câmera RTMP")
    exit()

# Obtém as propriedades do stream
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Configura a gravação do vídeo processado
output = cv2.VideoWriter('camera_processada.mp4', 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (frame_width, frame_height))

print("Conectado ao stream RTMP. Pressione 'q' para sair.")

# Adicione estas constantes no início do arquivo, após as importações
CLASSES = {
    0: 'person',  # classe que queremos detectar
}

# Modifique a parte de detecção no loop principal
while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Erro ao ler frame. Tentando reconectar...")
            video = cv2.VideoCapture(RTMP_URL)
            continue

        # Realiza a detecção com o YOLO
        resultados = modelo.track(frame, persist=True, classes=[0])  # Filtra apenas pessoas

        if resultados[0].boxes.id is not None:
            boxes = resultados[0].boxes.xyxy.cpu()
            track_ids = resultados[0].boxes.id.int().cpu().tolist()
            classes = resultados[0].boxes.cls.cpu().tolist()  # Obtém as classes
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) == 0:  # Verifica se é pessoa
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Desenha a caixa delimitadora
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Adiciona o ID e a classe
                    cv2.putText(frame, f'ID: {track_id} - Pessoa', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Atualiza o tracking
                    track = track_history[track_id]
                    track.append(float(y1))
                    
                    if len(track) > 2:
                        # Contagem de pessoas
                        if track[-2] >= LINHA_VIRTUAL and track[-1] < LINHA_VIRTUAL:
                            pessoas_entrando += 1
                        if track[-2] < LINHA_VIRTUAL and track[-1] >= LINHA_VIRTUAL:
                            pessoas_saindo += 1
                        
                        track_history[track_id] = track[-30:]

        # Desenha elementos visuais
        cv2.line(frame, (0, LINHA_VIRTUAL), (frame.shape[1], LINHA_VIRTUAL), (0, 255, 0), 2)
        cv2.putText(frame, f'Entrando: {pessoas_entrando}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Saindo: {pessoas_saindo}', (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Salva e mostra o frame
        output.write(frame)
        cv2.imshow('Camera RTMP', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erro: {str(e)}")
        continue

# Limpa recursos
video.release()
output.release()
cv2.destroyAllWindows()

