import cv2
import numpy as np


def detect_objects(image_path, output_path, config, weights, classes_file):

    def denoise_image(image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}",
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ---------------- LOAD IMAGE ----------------
    image = cv2.imread(image_path)

    image = denoise_image(image)

    Height, Width = image.shape[:2]
    scale = 0.00392

    # ---------------- LOAD CLASSES ----------------
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # ---------------- LOAD MODEL ----------------
    net = cv2.dnn.readNet(weights, config)
    output_layers = get_output_layers(net)

    # ---------------- MULTI-SCALE ----------------
    scales = [320, 416, 608]

    boxes = []
    confidences = []
    class_ids = []

    conf_threshold = 0.5
    nms_threshold = 0.4

    for size in scales:

        blob = cv2.dnn.blobFromImage(
            image, scale, (size, size), (0, 0, 0), True, crop=False
        )

        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # ---------------- NMS ----------------
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            i = i[0]
        except:
            pass

        x, y, w, h = boxes[i]

        draw_prediction(image, class_ids[i], confidences[i],
                        x, y, x + w, y + h)

    # ---------------- SAVE OUTPUT ----------------
    cv2.imwrite(output_path, image)