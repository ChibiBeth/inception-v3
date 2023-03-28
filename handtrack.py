import csv
import getopt
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_video_parts(video_path):
    parts = video_path.split(os.path.sep)
    # nombre de archivo de video
    filename = parts[3]
    # nombre de archivo de video sin extension
    filename_no_ext = filename.split('.')[0]
    # clase a la que pertenece el video
    classname = parts[2]
    # si el video procesado pertenece al grupo de test o train
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename


def hands_extraction(vid_dir, out_dir, resize=(299, 299)):
    data_file = []
    folders = ['train', 'test']
    seq_chek_folders = ['sequences', 'checkpoints']
    print(os.path.join(out_dir))
    # si el directorio output no existe, se crea
    if not os.path.exists(os.path.join(out_dir)):
        os.mkdir(os.path.join(out_dir))
        for folder in seq_chek_folders:
            if not os.path.exists(os.path.join(out_dir, folder)):
                os.mkdir(os.path.join(out_dir, folder))
    # se hace un recorrido en los directorios test y train
    for folder in folders:
        print(os.path.join(out_dir, folder))
        # si en el directorio de salida no existe el folder actual (test o train), se crea
        if not os.path.exists(os.path.join(out_dir, folder)):
            os.mkdir(os.path.join(out_dir, folder))
        print(folder)
        # se crea una lista de todas las carpetas en el directorio folder (test o train) que corresponden a las
        # clases (palabras)
        class_folders = glob.glob(os.path.join(vid_dir, folder, '*'))
        print(class_folders)
        # se recorre cada carpeta en el listado class_folders
        for vid_class in class_folders:
            # se crea una lista de todos los videos encontrados en la clase actual
            class_files = glob.glob(os.path.join(vid_class, '*'))

            # se analiza cada video encontrado
            for file_name in class_files:
                print(file_name)

                # se obtienen los datos relevantes del video actual
                video_parts = get_video_parts(file_name)

                # se almacenan los datos del video
                train_or_test, classname, filename_no_ext, filename = video_parts
                # se carga el video para su análisis
                cap = cv2.VideoCapture(file_name)

                i = 0

                # si la carpeta de clase no existe, se crea
                print(os.path.join(out_dir, train_or_test, classname))
                if not os.path.exists(os.path.join(out_dir, train_or_test, classname)):
                    os.mkdir(os.path.join(out_dir, train_or_test, classname))
                color = (0, 0, 0)
                img = np.full((299, 299, 3), color, np.uint8)

                # se utiliza la biblioteca mediapipe para la detección de las manos en los videos
                with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4) as hands:
                    nb_frames = 0
                    while cap.isOpened():
                        i += 1
                        ret, frame = cap.read()
                        if not ret:
                            print("Ignoring empty camera frame.")
                            # If loading a video, use 'break' instead of 'continue'.
                            break
                        frame = cv2.resize(frame, resize)

                        # BGR 2 RGB
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Flip on horizontal
                        image = cv2.flip(image, 1)

                        # Set flag
                        image.flags.writeable = False

                        # Detections
                        results = hands.process(image)

                        # Set flag to true
                        image.flags.writeable = True

                        # FONDO NEGRO
                        color = (0, 0, 0)
                        # IMAGEN DE 860x720 x3 canales
                        img = np.full((299, 299, 3), color, np.uint8)

                        # Detections

                        # Rendering results
                        if results.multi_hand_landmarks:
                            print(results.multi_hand_landmarks)
                            for num, hand in enumerate(results.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                                 circle_radius=1),
                                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                                 circle_radius=1),
                                                          )
                            # se escribe cada frame en el directorio de salida que le corresponda a su clase
                            cv2.imwrite(os.path.join(out_dir, train_or_test, classname,
                                                     '{}-{}.jpg'.format(filename_no_ext, str.rjust(str(i), 4, '0'))),
                                        img)
                            nb_frames = i
                        else:
                            i -= 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                # se guardan los datos relevantes de cada video procesado
                if nb_frames < 40:
                    for i in range(nb_frames, 42, 1):
                        cv2.imwrite(os.path.join(out_dir, train_or_test, classname,
                                                 '{}_{}.jpg'.format(filename_no_ext, str.rjust(str(i), 4, '0'))), img)
                    nb_frames = 41

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                cap.release()
                cv2.destroyAllWindows()
    # se escriben los datos relevantes de los videos procesados en el archivo data_file.csv
    with open(os.path.join(out_dir, 'data_file.csv'), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def main(argv):
    # se introduce el directorio donde se encuentran todos los videos
    # todo lo procesado se guarda en la carpeta output
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    for opt, arg in opts:
        if opt == '-h':
            print('handtrack.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if inputfile is not '' and outputfile is not '':
        hands_extraction(inputfile, outputfile)
    else:
        print('handtrack.py -i <inputfile> -o <outputfile>')
        sys.exit()


if __name__ == '__main__':
    main(sys.argv[1:])
