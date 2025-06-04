import cv2
import mediapipe as mp
import math
import csv 
from datetime import datetime

# Inicializa MediaPipe Face_Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Definir los indices de referencia labial
OUTER_LIPS_IDXS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 62, 96, 89, 180, 85, 16, 315, 404, 319, 325, 292]
INNER_LIPS_IDXS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 76, 191, 81, 82, 13, 312, 311, 415, 310]

# Índices de referencia específicos 
UPPER_LIP_PT_IDX = 13
LOWER_LIP_PT_IDX = 14
LEFT_MOUTH_CORNER_IDX = 61
RIGHT_MOUTH_CORNER_IDX = 291

# Clasificación del estado de la boca (es necesario ajustarlos en función de la observación)
MOUTH_CLOSED_THRESHOLD = 0.04
MOUTH_OPEN_THRESHOLD = 0.1

# CSV Setup
CSV_FILENAME = "lip_features_log.csv"
CSV_HEADER = [
    "timestamp", "vertical_opening", "horizontal_width",
    "upper_lip_x", "upper_lip_y", "lower_lip_x", "lower_lip_y",
    "left_corner_x", "left_corner_y", "right_corner_x", "right_corner_y"
]

# Inicializa la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Escribir el CSV
try:
    csv_file = open(CSV_FILENAME, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(CSV_HEADER)
except IOError as e:
    print(f"Error opening CSV file: {e}")
    exit()


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar puntos de referencia para los labios
                for idx in OUTER_LIPS_IDXS:
                    if 0 <= idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        cx, cy = int(landmark.x * img_w), int(landmark.y * img_h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                for idx in INNER_LIPS_IDXS:
                    if 0 <= idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        cx, cy = int(landmark.x * img_w), int(landmark.y * img_h)
                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                
                required_indices = [UPPER_LIP_PT_IDX, LOWER_LIP_PT_IDX, LEFT_MOUTH_CORNER_IDX, RIGHT_MOUTH_CORNER_IDX]
                if all(0 <= idx < len(face_landmarks.landmark) for idx in required_indices):
                    upper_lip_pt = face_landmarks.landmark[UPPER_LIP_PT_IDX]
                    lower_lip_pt = face_landmarks.landmark[LOWER_LIP_PT_IDX]
                    left_mouth_corner = face_landmarks.landmark[LEFT_MOUTH_CORNER_IDX]
                    right_mouth_corner = face_landmarks.landmark[RIGHT_MOUTH_CORNER_IDX]

                    vertical_opening = math.hypot(upper_lip_pt.x - lower_lip_pt.x, 
                                                  upper_lip_pt.y - lower_lip_pt.y)
                    horizontal_width = math.hypot(right_mouth_corner.x - left_mouth_corner.x,
                                                   right_mouth_corner.y - left_mouth_corner.y)
                    
                    timestamp = datetime.now().isoformat()
                    try:
                        log_row = [
                            timestamp, vertical_opening, horizontal_width,
                            upper_lip_pt.x, upper_lip_pt.y, lower_lip_pt.x, lower_lip_pt.y,
                            left_mouth_corner.x, left_mouth_corner.y, right_mouth_corner.x, right_mouth_corner.y
                        ]
                        csv_writer.writerow(log_row)
                    except Exception as e:
                        print(f"Error writing to CSV: {e}")

                    # Características de la pantalla
                    cv2.putText(frame, f"Open: {vertical_opening:.4f}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Width: {horizontal_width:.4f}", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Determinar el estado de la boca
                    if vertical_opening < MOUTH_CLOSED_THRESHOLD:
                        mouth_state_str = "Closed"
                    elif vertical_opening > MOUTH_OPEN_THRESHOLD:
                        mouth_state_str = "Open"
                    else:
                        mouth_state_str = "Neutral"
                    
                    cv2.putText(frame, f"Mouth State: {mouth_state_str}", (20, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                else:
                    # Mostrar advertencia y estado N/D si no se detectan puntos de referencia
                    cv2.putText(frame, "Landmarks not detected", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Mouth State: N/A", (20, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera Feed with Lip Landmarks", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
if 'csv_file' in locals() and not csv_file.closed:
    csv_file.close()
    print(f"CSV log saved to {CSV_FILENAME}")
cv2.destroyAllWindows()
print("Camera released and windows closed.")
