import cv2
import os
import numpy as np 

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import keras
import suduko_solver 

from tensorflow.keras.models import load_model
model = load_model('digitrecon.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def printing(arr):
	for i in range(N):
		for j in range(N):
			print(arr[i][j], end = " ")
		print()





font_color = (0, 127, 255)
font_path = 'FreeMono.ttf'

def show_image(image):
    plt.figure(figsize=(10,10))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
def preprocess(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # inverted = cv2.bitwise_not(thresh, 0)
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # dilated = cv2.dilate(morph, kernel, iterations=1)
    return thresh
def biggestcontours(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, peri * 0.02, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(points):
    points =points.reshape((4,2))
    mypoints = np.zeros((4,1,2), dtype=np.int32)
    add = points.sum(1)
    mypoints[0] = points[np.argmin(add)]
    mypoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    mypoints[1] = points[np.argmin(diff)]
    mypoints[2] = points[np.argmax(diff)]
    return mypoints

def splitboxes(image):
    row = np.vsplit(image,9)
    boxes = []
    for i in row:
        column = np.hsplit(i,9)
        for box in column:
            boxes.append(box)
    return boxes
def resize_keep_aspect(img, size=630):
    old_height, old_width = img.shape[:2]
    if img.shape[0] >= size:
        aspect_ratio = size / float(old_height)
        dim = (int(old_width * aspect_ratio), size)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    elif img.shape[1] >= size:
        aspect_ratio = size / float(old_width)
        dim = (size, int(old_height * aspect_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    return img
def get_grid_lines(img, length=12):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    return vertical, horizontal

def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners
def transform(pts, img):  # TODO: Spline transform, remove this
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped

def stitch_img(img_arr, img_dims):
    result = Image.new('RGB' if len(img_arr[0].shape) > 2 else 'L', img_dims)
    box = [0, 0]
    for img in img_arr:
        pil_img = Image.fromarray(img)
        result.paste(pil_img, tuple(box))
        if box[0] + img.shape[1] >= result.size[1]:
            box[0] = 0
            box[1] += img.shape[0]
        else:
            box[0] += img.shape[1]
    return np.array(result)
def get_grid_lines(img, length=12):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    return vertical, horizontal
def create_grid_mask(vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, 1, 1, 11, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(im, pts):
        im = np.copy(im)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(im, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return im

    lines = draw_lines(grid, pts)
    # mask = cv2.bitwise_not(lines)
    return lines
def extract_digits(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    # Reversing contours list to loop with y coord ascending, and removing small bits of noise
    contours_denoise = [i for i in contours[::-1] if cv2.contourArea(i) > img_area * .0005]
    _, y_compare, _, _ = cv2.boundingRect(contours_denoise[0])
    digits = []
    row = []

    for i in contours_denoise:
        x, y, w, h = cv2.boundingRect(i)
        cropped = img[y:y + h, x:x + w]
        if y - y_compare > img.shape[1] // 40:
            row = [i[0] for i in sorted(row, key=lambda x: x[1])]
            for j in row:
                digits.append(j)
            row = []
        row.append((cropped, x))
        y_compare = y
    # Last loop doesn't add row
    row = [i[0] for i in sorted(row, key=lambda x: x[1])]
    for i in row:
        digits.append(i)

    return digits
def subdivide(img, divisions=9):
    height, _ = img.shape[:2]
    box = height // divisions
    if len(img.shape) > 2:
        subdivided = img.reshape(height // box, box, -1, box, 3).swapaxes(1, 2).reshape(-1, box, box, 3)
    else:
        subdivided = img.reshape(height // box, box, -1, box).swapaxes(1, 2).reshape(-1, box, box)
    return [i for i in subdivided]

def add_border(img_arr):
    digits = []
    for i in img_arr:
        crop_h, crop_w = i.shape[:2]
        try:
            pad_h = int(crop_h / 1.75)
            pad_w = (crop_h - crop_w) + pad_h
            pad_h //= 2
            pad_w //= 2
            border = cv2.copyMakeBorder(i, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            digits.append(border)
        except cv2.error:
            continue
    dims = (digits[0].shape[0],) * 2
    digits_square = [cv2.resize(i, dims, interpolation=cv2.INTER_NEAREST) for i in digits]
    return digits_square

def add_zeros(sorted_arr, subd_arr):
    h, w = sorted_arr[0].shape
    print(sorted_arr[0].shape)
    puzzle_template = np.zeros((81, h, w), dtype=np.uint8)
    sorted_arr_idx = 0
    for i, j in enumerate(subd_arr):
        if np.sum(j) < 9000:
            zero = np.zeros((h, w), dtype=np.uint8)
            puzzle_template[i] = zero
        else:
            print(i)
            puzzle_template[i] = sorted_arr[sorted_arr_idx]
            sorted_arr_idx += 1
        # print(i)
    return puzzle_template

def img_to_array(img_arr, img_dims):
    predictions = []
    for i in img_arr:
        resized = cv2.resize(i, (img_dims, img_dims), interpolation=cv2.INTER_LANCZOS4)
        if np.sum(resized) == 0:
            predictions.append(0)
            continue
        array = np.array([resized])
        reshaped = array.reshape(array.shape[0], img_dims, img_dims, 1)
        flt = reshaped.astype('float32')
        flt /= 255
        prediction = model.predict(flt)
        class_index = np.argmax(prediction, axis=-1)
        probability = np.amax(prediction)
        # print(class_index, probability)
        if probability > 0.75:
            predictions.append(class_index[0])
            # OCR predicts from 0-8, changing it to 1-9
        else:
            predictions.append(0)
    puzzle = np.array(predictions).reshape((9, 9))
    return puzzle

def put_solution(img_arr, soln_arr, unsolved_arr, font_color, font_path):
    solutions = np.array(soln_arr).reshape(81)
    unsolveds = np.array(unsolved_arr).reshape(81)
    paired = list((zip(solutions, unsolveds, img_arr)))
    img_solved = []
    for solution, unsolved, img in paired:
        if unsolved == 0:
            img_solved.append(img)
            continue
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        fnt = ImageFont.truetype(font_path, img_h)
        font_w, font_h = draw.textsize(str(solution), font=fnt)
        draw.text(((img_w - font_w) / 2, (img_h - font_h) / 2 - img_h // 10), str(solution),
                  fill=(font_color if len(img.shape) > 2 else 0), font=fnt)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_solved.append(cv2_img)
    return img_solved

def get_predictions(boxes,model):
    result = []
    for img in boxes:
        image = np.asarray(img)
        image = image[4:image.shape[0]-4,4:image.shape[1]-4]
        img = cv2.resize(image, (28,28))
#         show_image(img)
        if np.sum(img) < 12000:
            result.append(0)
            continue
        img = img/255
        img = img.reshape(1,28,28,1)
        
        prediction = model.predict(img)
        class_index = np.argmax(prediction, axis=-1)
        probability = np.amax(prediction)
        # print(class_index, probability)
        if probability > 0.75:
            result.append(class_index[0])
        else:
            result.append(0)
    return result

def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img

camera = cv2.VideoCapture(0)
y = 0
while 1:
    path = ['1_74imtxlSUemc7meOg-o0UQ.jpeg', 'WIN_20170813_00_09_42_Pro.jpg', 'download.png']
    try:
        sid, image  = camera.read()
        y = (y + 1) % 3
        image = resize_keep_aspect(image)
        imt = image
        img_threshold = preprocess(image)
        img_contours = image.copy()
        img_big = image.copy()
        imt = img_big
        corners = get_corners(img_threshold)
        warped = transform(corners, img_big)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(warped, 255, 1, 1, 11, 2)
        boxes = splitboxes(thresh)
        cells = [np.pad(np.ones((7, 7), np.uint8) * 255, (1, 1), 'constant', constant_values=(0, 0)) for _ in range(81)]
        grid = stitch_img(cells, (81, 81))
        template = cv2.resize(grid, (warped.shape[0],) * 2, interpolation=cv2.INTER_NEAREST)

        vertical_lines, horizontal_lines = get_grid_lines(warped)
        mask = create_grid_mask(vertical_lines, horizontal_lines)
        res = cv2.matchTemplate(mask, template, cv2.TM_CCORR_NORMED)
        threshold = .55
        loc = 1
        if loc == 0:
            raise ValueError('Grid template not matched')
        else:
            a = get_predictions(boxes, model)
            b = [[],[],[],[],[],[],[],[],[]]
            for i in range(0,9):
                for j in range(0,9):
                    b[i].append(a[i * 9 + j])
            print(b)
            number = np.asarray(b)
            posarray = np.where(number > 0,0,1)
            suduko_solver.solve_sudoku(b) # Solve it
            
            if not suduko_solver.all_board_non_zero(b):
                raise ValueError('in') 
            font_color = (0, 127, 255)
            c = b*posarray
            font_path = 'FreeMono.ttf'
            warped_img = transform(corners, imt)
            subd = subdivide(warped_img)
            subd_soln = put_solution(subd, b, c, font_color, font_path)
            warped_soln = stitch_img(subd_soln, (warped_img.shape[0], warped_img.shape[1]))
            warped_inverse = inverse_perspective(warped_soln, image, np.array(corners))
            cv2.imshow('ima', warped_inverse)
            cv2.waitKey(1)
        
    except Exception as e:
        cv2.imshow('frame', image)
        print(e)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue