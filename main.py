import cv2
import numpy as np

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

min_size = 128 # Minimum eligible hand size, for removing false detected contour
in_size = 64 # Size of resized image
in_data = None # Buffer for input image of the model

# Rescale Image
def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Find Contour
def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

# Find Biggest Contour
def max_contour(contour_list):
    if len(contour_list) == 0:
        return None

    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

    return contour_list[max_i]

# Calculate Centroid from given Contour
def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

# Calculate Histogram of the hands
def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

# Image Segmentation
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))

    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 220, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)

# Detect Hand Image
def manage_image_opr(frame, hand_hist):
    ## Segmentation ##
    hist_mask_image = hist_masking(frame, hand_hist)

    ## Find Best Contour ##
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    if max_cont is not None:
        ## Find Border Points ##
        hulls = cv2.convexHull(max_cont, returnPoints=True)
        p1 = np.squeeze(hulls.min(axis=0))
        p2 = np.squeeze(hulls.max(axis=0))

        ## Measure Resize Shape ##
        global in_data, in_size, min_size
        p1x = 0 if p1[1] - 10 < 0 else p1[1] - 10
        p1y = 0 if p1[0] - 10 < 0 else p1[0] - 10
        p2x = 0 if p2[1] + 10 < 0 else p2[1] + 10
        p2y = 0 if p2[0] + 10 < 0 else p2[0] + 10

        # Check if max contour size is more than minimum size
        if p2x - p1x >= min_size or p2y - p2x >= min_size:
            hand_img = frame[p1x:p2x,p1y:p2y, :]
            if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                h_size = 0
                w_size = 0
            elif hand_img.shape[0] > hand_img.shape[1]:
                h_size = in_size
                w_size = hand_img.shape[1] * in_size // hand_img.shape[0]
            elif hand_img.shape[0] < hand_img.shape[1]:
                h_size = hand_img.shape[0] * in_size // hand_img.shape[1]
                w_size = in_size
            else:
                h_size = in_size
                w_size = in_size

            if h_size > 0 and w_size > 0:
                ## Resize Hand Image ##
                hand_img = cv2.resize(hand_img, (w_size, h_size))
                
                ## Pad Hand Image ##
                in_data = np.zeros((in_size,in_size,3))
                in_data[-hand_img.shape[0]:,-hand_img.shape[1]:,:] = hand_img

                # Draw Detected Hand Frame on Top Left Corner
                frame[:64,:64,:] = in_data

            ## Draw Bounding Box ##
            cv2.rectangle(frame,(p1[0] - 10, p1[1] - 10),(p2[0] + 10, p2[1] + 10),(0,255,0),2)


    return frame

# Draw Rectangle to Locate Position for Histogram Measurement
def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()

        # Start / Stop Detection when 'z' pressed
        if pressed_key & 0xFF == ord('z'):
            if is_hand_hist_created:
                is_hand_hist_created = False
            else:
                is_hand_hist_created = True
                hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            frame = manage_image_opr(frame, hand_hist)
        else:
            frame = draw_rect(frame)

        cv2.imshow("FunTorch", rescale_frame(frame))

        # Close if ESC pressed
        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()
