import cv2
import time
import numpy as np



def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

def identify(video):

    eventos = []

    # Cores utilizadas nos quadrados das detecções
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    # Listas que irão guardar as classes possíveis e outra que guardará apenas classes que desejamos trabalhar
    class_names = []
    with open("coconames", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    util_class_names = []
    with open("utilnames", "r") as f:
        util_class_names = [cname.strip() for cname in f.readlines()]

    # Redes pré-treinadas YOLO
    #net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")


    model = cv2.dnn.DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255) # da CFG

    tempo = 0

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('frames per second =',fps)

    minutes = 1
    seconds = 0
    frame_id = int(fps*(0*60 + 3))
    print('frame id =',frame_id)

    # Enquanto há frames do vídeo para processar
    i = 0

    ret, frame = cap.read()
    while ret:
        #print(i, int(fps*(1*60 + 0)))

        # Demarca FPS
        start = time.time()

        # Detecção
        classes, scores, boxes = model.detect(frame, 0.1, 0.2)

        tempo += time.time() - start
        end = time.time()


        # Percorrer as detecções e imprimí-las na tela junto do frame correspondente
        for (classid, score, box) in zip(classes, scores, boxes):
            
            # Analisa se é uma classe desejada
            if class_names[classid] in util_class_names and score >= 0.5:

                
                if(not eventos or eventos[-1][0] < convert(tempo)):
                    eventos.append([convert(tempo)])

                for evento in eventos:
                    tempo_evento = evento[0]
                    if tempo_evento == convert(tempo) and class_names[classid] not in evento:
                        evento.append(class_names[classid])

                color = COLORS[int(classid) % len(COLORS)]

                #print(f"Tag: {class_names[classid]} \tPrecisao: {score:.2f} \tTempo: {convert(tempo)}")

                label = f"{class_names[classid]} : {100 * score:.2f}%"

                cv2.rectangle(frame, box, color, 2)

                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.imshow("VigIA", frame)

        # Espera de resposta para finalizar janela
        if cv2.waitKey(1) == 27:
            break

        i += 1

        # Escreve o que pegou naquele minuto
        if(i % int(fps*(1*60 + 0)) == 0):
            f = open("events.txt", "a")
            for item in eventos:
                string = ''
                for element in item:
                    string = string + str(element) + ';'
                string = string + '\n'
                f.write(string)
            f.close()
            eventos = []

        ret, frame = cap.read()

    # Libera e destrói janelas
    cap.release()
    cv2.destroyAllWindows()
    
    return eventos





if __name__ == '__main__':

    video = "video-test3.mp4"

    eventos = identify(video)

    # Escreve o que sobrou
    f = open("events.txt", "a")
    for item in eventos:
        string = ''
        for element in item:
            string = string + str(element) + ';'
        string = string + '\n'
        f.write(string)
    f.close()