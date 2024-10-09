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
                # Obtenha as coordenadas dos pontos da boca
                left_corner = face_landmarks.landmark[48]  # Ponto 48 (canto esquerdo da boca)
                right_corner = face_landmarks.landmark[54]  # Ponto 54 (canto direito da boca)
                upper_lip = face_landmarks.landmark[51]     # Ponto 51 (meio da boca superior)
                lower_lip = face_landmarks.landmark[57]     # Ponto 57 (meio da boca inferior)

                # Obtenha as coordenadas dos olhos e do queixo
                left_eye = face_landmarks.landmark[33]      # Ponto 33 (canto esquerdo do olho)
                right_eye = face_landmarks.landmark[133]     # Ponto 133 (canto direito do olho)
                chin = face_landmarks.landmark[152]          # Ponto 152 (queixo)

                # Calcular coordenadas
                h, w, _ = img.shape
                left_x = int(left_corner.x * w)
                right_x = int(right_corner.x * w)
                upper_y = int(upper_lip.y * h)
                lower_y = int(lower_lip.y * h)
                left_eye_x = int(left_eye.x * w)
                right_eye_x = int(right_eye.x * w)
                chin_y = int(chin.y * h)

                # Verifica se a boca está sorrindo ou triste
                mouth_width = right_x - left_x
                mouth_height = lower_y - upper_y
                
                # Debug: imprima os valores
                print(f"Mouth Width: {mouth_width}, Mouth Height: {mouth_height}")

                # Desenhe os pontos de referência
                cv2.circle(img, (left_x, upper_y), 3, (0, 255, 0), -1)  # Canto esquerdo da boca
                cv2.circle(img, (right_x, upper_y), 3, (0, 255, 0), -1)  # Canto direito da boca
                cv2.circle(img, (left_eye_x, int(left_eye.y * h)), 3, (0, 0, 255), -1)  # Olho esquerdo
                cv2.circle(img, (right_eye_x, int(right_eye.y * h)), 3, (0, 0, 255), -1)  # Olho direito
                cv2.circle(img, (int(chin.x * w), chin_y), 3, (255, 0, 0), -1)  # Queixo

                # Definindo limites mais adequados
                if mouth_height > 20 and mouth_width > 60:  # Sorriso
                    cv2.putText(img, 'Feliz', (left_x, upper_y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.line(img, (left_x, lower_y), (right_x, lower_y), (255, 0, 0), 2)
                else:  # Triste
                    cv2.putText(img, 'Triste', (left_x, upper_y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.line(img, (left_x, lower_y), (right_x, lower_y), (255, 0, 0), 2)

        # Exibe a imagem com os rostos detectados
        cv2.imshow("imagem", img)

        # Sai do loop ao pressionar a tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera o vídeo e fecha as janelas
    video.release()
    cv2.destroyAllWindows()

detect_facial_expression()
