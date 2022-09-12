import cv2
import numpy as np

def apply_yolo_object_detection(image_to_process):
    """
    Распознавание и определение координат объектов на изображении
    :param image_to_process: исходное изображение
    :return: изображение с размеченными объектами и подписми к ним
    """
    height, width, deoth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608,608), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forfard(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    #Запуск поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * width)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * width)
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_score.append(float(class_scores))
    #Выборка
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        bo_index = box_index[0]
