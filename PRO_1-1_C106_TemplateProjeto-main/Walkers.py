import cv2

# Caminho para o arquivo xml do classificador em cascata
cascade_path = 'caminho/para/haarcascade_file.xml'

# Inicializar o classificador de corpos
body_classifier = cv2.CascadeClassifier()

# Carregar o classificador
if not body_classifier.load(cv2.samples.findFile(cascade_path)):
    print("Erro ao carregar o classificador.")
    exit()
    
# Supondo que você tenha uma lista de corpos onde cada corpo é uma tupla (x, y, w, h)
bodies = [(100, 100, 50, 50), (200, 150, 70, 60), (300, 200, 80, 70)]

# Função para desenhar retângulo em cada corpo
def draw_rectangles(frame, bodies):
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Função principal para exibir o quadro com retângulos
def main():
    # Inicializar a captura de vídeo (substitua pela sua própria captura de vídeo)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar corpos usando o classificador em cascata
        detected_bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Desenhar retângulos em torno de corpos detectados
        draw_rectangles(frame, detected_bodies)
        
        # Exibir o quadro com os retângulos
        cv2.imshow('Frame', frame)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar a captura de vídeo e fechar todas as janelas abertas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()