import cv2


def draw_boxes(image, boxes, scores, labels, colors, classes):
    b = boxes
    l = labels
    s = scores
    print(classes)
    print(b)
    print(l)
    print(s)
    class_id = int(l)
    class_name = classes[class_id]

    xmin, ymin, xmax, ymax = list(map(int, b))
    score = '{:.4f}'.format(s)
    color = colors
    label = '-'.join([class_name, score])

    ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
    cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
