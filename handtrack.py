import getopt
import os
import glob
import sys

import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def max_detection(file_name, vid_dir, padding=0, resize=False):
    cap = cv2.VideoCapture(os.path.join(vid_dir, f'{file_name}.mp4'))

    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera: max_detection fase")
                break
            if resize:
                frame = cv2.resize(frame, resize)
            (h, w, c) = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    for lm in hand.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
    return (x_max + padding, y_max + padding, max(x_min - padding, 0), max(y_min - padding, 0))


def get_video_parts(video_path):
    parts = video_path.split(os.path.sep)
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    classname = parts[2]
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename


def hands_extraction(vid_dir, out_dir, resize=(399, 299)):
    # file_name="WIFI"
    folders = ['train', 'test']
    list_paths = []
    #se agrega creacion de folders sequences y checkpoints que las demas partes usan
    if not os.path.exists(os.path.join(out_dir, 'sequences')):
        os.mkdir(os.path.join(out_dir, 'sequences'))
    if not os.path.exists(os.path.join(out_dir, 'checkpoints')):
        os.mkdir(os.path.join(out_dir, 'checkpoints'))
    for subdir, dirs, files in os.walk(vid_dir):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            list_paths.append(filepath)

    for folder in folders:
        print(folder)
        class_folders = glob.glob(os.path.join(vid_dir, folder, '*'))
        print(class_folders)
        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*'))

            for file_name in class_files:
                print(file_name)
                # file_path, file = file_name.split('/')
                # name, ext = file.split('.')

                video_parts = get_video_parts(file_name)

                train_or_test, classname, filename_no_ext, filename = video_parts

                cap = cv2.VideoCapture(file_name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                output_image_array = []
                fourcc = 'DIVX'
                # TODO: detectar los maximos y minimos
                # x_max, y_max, x_min, y_min = max_detection(file_name, vid_dir, resize=resize)
                # print((x_max,y_max,x_min,y_min))

                # cap = cv2.VideoCapture(0)

                i = 0
                print(os.path.join(out_dir))
                if not os.path.exists(os.path.join(out_dir)):
                    os.mkdir(os.path.join(out_dir))
                print(os.path.join(out_dir, train_or_test))
                if not os.path.exists(os.path.join(out_dir, train_or_test)):
                    os.mkdir(os.path.join(out_dir, train_or_test))
                print(os.path.join(out_dir, train_or_test, classname))
                if not os.path.exists(os.path.join(out_dir, train_or_test, classname)):
                    os.mkdir(os.path.join(out_dir, train_or_test, classname))

                with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
                    while cap.isOpened():
                        i += 1
                        ret, frame = cap.read()
                        if not ret:
                            print("Ignoring empty camera frame.")
                            # If loading a video, use 'break' instead of 'continue'.
                            break
                        (h, w, c) = frame.shape
                        frame = cv2.resize(frame, resize)
                        # print((h,w,c))
                        # if i==1:
                        #    out = cv2.VideoWriter('project.avi',fourcc, fps, (h,w))

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

                            for num, hand in enumerate(results.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS,
                                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                                 circle_radius=1),
                                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                                 circle_radius=1),
                                                          )

                            cv2.imwrite(os.path.join(out_dir, train_or_test, classname,
                                                     '{}{}.jpg'.format(filename_no_ext, str.rjust(str(i), 6, '0'))),
                                        img)
                        else:
                            i -= 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                # create_video.create_video(img_dir=f"{out_dir}/{name}", fps=int(fps / 3), fourcc=fourcc,
                #                           video=f"{out_dir}/{name}.avi")

                # out.release()
                cap.release()
                cv2.destroyAllWindows()


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
