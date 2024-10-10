import cv2
import mediapipe as mp


def detect_facial_expression():
    # Inicialização da captura de vídeo
    video = cv2.VideoCapture(0)

    # Inicialização do MediaPipe Face Mesh
    mpFaceMesh = mp.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

    # Inicialização dos utilitários de desenho
    mpDraw = mp.solutions.drawing_utils

    # Lista de índices dos pontos da boca e do rosto
    mouth_indices = list(range(0, 18))  # Pontos 0 a 17 para a boca
    face_indices = list(range(0, 468))   # Todos os pontos do rosto

    while True:
        check, img = video.read()
        
        # Verifica se a imagem foi capturada corretamente
        if not check:
            print("Erro ao capturar a imagem.")
            break

        # Converte a imagem de BGR para RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem para detectar a malha facial
        results = face_mesh.process(imgRGB)

        # Desenha a malha facial e determina a posição da boca
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenha todos os pontos do rosto
                for index in face_indices:
                    landmark = face_landmarks.landmark[index]
                    h, w, _ = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)  # Desenha um círculo em todos os pontos do rosto

                # Obtenha as coordenadas dos pontos da boca
                left_corner = face_landmarks.landmark[48]  # Ponto 48 (canto esquerdo da boca)
                right_corner = face_landmarks.landmark[54]  # Ponto 54 (canto direito da boca)
                upper_lip = face_landmarks.landmark[13]  # Ponto 13 (meio do lábio superior)
                lower_lip = face_landmarks.landmark[14]  # Ponto 14 (meio do lábio inferior)

                # Calcular coordenadas
                h, w, _ = img.shape
                left_x = int(left_corner.x * w)
                right_x = int(right_corner.x * w)
                upper_y = int(upper_lip.y * h)
                lower_y = int(lower_lip.y * h)

                # Desenhe a linha da boca
                cv2.line(img, (left_x, int(left_corner.y * h)), (right_x, int(right_corner.y * h)), (255, 0, 0), 2)

                # Verifica se a boca está sorrindo ou triste com base na diferença vertical entre o lábio superior e inferior
                mouth_height = lower_y - upper_y

                if mouth_height > 10:  # Ajuste fino: agora precisa abrir menos a boca para detectar sorriso
                    cv2.putText(img, 'Feliz', (left_x, int(left_corner.y * h) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:  # Se o lábio inferior está mais próximo do superior (triste)
                    cv2.putText(img, 'Triste', (left_x, int(left_corner.y * h) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Exibe a imagem com os rostos detectados
        cv2.imshow("imagem", img)
        cv2.waitKey(1)

        # Sai do loop ao pressionar a tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera o vídeo e fecha as janelas
    video.release()
    cv2.destroyAllWindows()
