def normalise_landmarks(landmarks):
    """
    :param landmarks: [[x1,y1], [x2,y2], ...]
    :return: normalised list
    """
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    thumb_x, thumb_y = landmarks[1][0], landmarks[1][1]
    scale_factor = (((thumb_x - base_x) ** 2) + ((thumb_y - base_y) ** 2)) ** 0.5



    for num in range(len(landmarks)):
        landmarks[num][0] -= base_x
        landmarks[num][0] /= scale_factor

        landmarks[num][1] -= base_y
        landmarks[num][1] /= scale_factor

    return landmarks