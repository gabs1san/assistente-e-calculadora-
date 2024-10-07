import cv2
import mediapipe as mp

# Iniciar captura de vídeo
video = cv2.VideoCapture(0)

# Configuração do MediaPipe para detecção de mãos
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2)

mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks

    h, w, _ = img.shape
    all_points = []  # Lista para armazenar os pontos de todas as mãos

    # Se detectar as mãos, desenha as landmarks e armazena os pontos
    if handsPoints:
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            pontos = []
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))
            all_points.append(pontos)

    dedos = [8, 12, 16, 20]  # Pontos que representam as pontas dos dedos
    Ccontador_mao1 = 0  # Contador para a primeira mão
    Ccontador_mao2 = 0  # Contador para a segunda mão (se detectada)

    # Contagem de dedos para a mão 1 (se houver pontos detectados)
    if len(all_points) > 0:
        pontos_mao1 = all_points[0]
        
        # Identificando se a primeira mão é a mão direita ou esquerda
        if pontos_mao1[4][0] < pontos_mao1[2][0]:  # Mão direita (polegar à esquerda)
            if pontos_mao1[4][0] < pontos_mao1[2][0]:  # Dedão (mão direita)
                Ccontador_mao1 += 1
            for x in dedos:
                if pontos_mao1[x][1] < pontos_mao1[x - 2][1]:
                    Ccontador_mao1 += 1
        else:  # Mão esquerda (polegar à direita)
            if pontos_mao1[4][0] > pontos_mao1[2][0]:  # Dedão (mão esquerda)
                Ccontador_mao1 += 1
            for x in dedos:
                if pontos_mao1[x][1] < pontos_mao1[x - 2][1]:
                    Ccontador_mao1 += 1

    # Contagem de dedos para a mão 2 (se houver uma segunda mão detectada)
    if len(all_points) > 1:
        pontos_mao2 = all_points[1]
        
        # Identificando se a segunda mão é a mão direita ou esquerda
        if pontos_mao2[4][0] < pontos_mao2[2][0]:  # Mão direita (polegar à esquerda)
            if pontos_mao2[4][0] < pontos_mao2[2][0]:  # Dedão (mão direita)
                Ccontador_mao2 += 1
            for x in dedos:
                if pontos_mao2[x][1] < pontos_mao2[x - 2][1]:
                    Ccontador_mao2 += 1
        else:  # Mão esquerda (polegar à direita)
            if pontos_mao2[4][0] > pontos_mao2[2][0]:  # Dedão (mão esquerda)
                Ccontador_mao2 += 1
            for x in dedos:
                if pontos_mao2[x][1] < pontos_mao2[x - 2][1]:
                    Ccontador_mao2 += 1

    # Mostrar o contador de dedos na tela para cada mão
    cv2.putText(img, f'Mao 1: {Ccontador_mao1}', (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.putText(img, f'Mao 2: {Ccontador_mao2}', (50, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 5)

    # Função de calculadora: Soma dos dedos das duas mãos
    soma = Ccontador_mao1 + Ccontador_mao2
    cv2.putText(img, f'Soma: {soma}', (50, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 5)

    # Mostrar a imagem na tela
    cv2.imshow("Imagem", img)
    cv2.waitKey(1)

     # Sai do loop ao pressionar a tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Libera o vídeo e fecha as janelas
video.release()
cv2.destroyAllWindows()