from argparse import ArgumentParser
from pathlib import Path
from random import sample
from shutil import copyfile

import cv2 as cv
import numpy as np
from PIL import Image
from screeninfo import get_monitors


def window_size(scale: float) -> tuple:
    """ Returns resolution of the monitor """
    monitor = get_monitors()[0]
    w = monitor.width
    h = monitor.height
    return round(w * scale), round(h * scale)


# Initialize maximum size of a window (updated in func load_image())
window_width, window_height = window_size(0.93)

# Empty helper arrays for storing image in different states
orig_img = np.empty((window_height, window_width))
thumb_img = np.empty((window_height, window_width))
thumb_img_clean = np.empty((window_height, window_width))
zoomed_img_clean = np.empty((window_height, window_width))

# Initialize scaling ratio (updated in func run_label_tool())
ratio = orig_img.shape[0] / window_height

# Store mouse last position
mouse_x, mouse_y = -1, -1
# ... and position before zooming (shift vector from source image origin to zoomed area origin)
shift_x, shift_y = -1, -1

# Initial circle radius in preview
circle_radius = round(0.04 * window_height)

# Current object class (for labeling different objects in same image)
object_class = 0  # 0 for bolt, 1 for lamp

# Colors of circles for different objects
object_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0)
}

# Image (Path): list of circle labels using original image coordinates (orig_x, orig_y, orig_r, object_class) mapping
image_circles_dict = {}

# Index of current image for images rewinding
current_image = 0

# Initial zoom state and zoom factor
zoomed = False
zoom = 5


def mouse_event(event, x, y, flags, param):
    """ Callback function for mouse handling """
    global mouse_x, mouse_y, shift_x, shift_y, thumb_img, zoomed, circle_radius, zoomed_img_clean

    # Store current mouse coordinates for every event occurred
    mouse_x, mouse_y = x, y

    if event == cv.EVENT_MOUSEWHEEL:
        # Forward motion (zoom)
        if flags > 0:
            if not zoomed:
                # Zoom image with zoom factor
                h, w = orig_img.shape[:2]
                zoom_y, zoom_x = round(h / zoom), round(w / zoom)
                # Convert from thumbnail coordinates to source image coordinates
                orig_y, orig_x = round(y * ratio), round(x * ratio)
                # Store coordinates of zoomed image origin (clip to boundaries)
                shift_y = np.clip(orig_y - round(zoom_y / 2), 0, h - zoom_y)
                shift_x = np.clip(orig_x - round(zoom_x / 2), 0, w - zoom_x)
                # New preview image, sub-matrix of original image (clip to boundaries)
                thumb_img = orig_img[shift_y: np.clip(orig_y + round(zoom_y / 2), zoom_y, h),
                                     shift_x: np.clip(orig_x + round(zoom_x / 2), zoom_x, w)]
                # Choose resize interpolation better suited for enlarging or shrinking
                if thumb_img.shape[0] < window_height:
                    inter = cv.INTER_CUBIC
                else:
                    inter = cv.INTER_AREA
                # Resize zoomed sub-image to fit window
                zoomed_img_clean = cv.resize(thumb_img, (window_width, window_height), interpolation=inter)
                thumb_img = zoomed_img_clean.copy()
                # Zoom circle radius
                circle_radius = round(circle_radius * zoom)
                # Set zoomed state
                zoomed = True
                # Repaint circles after zooming (refresh image)
                repaint_circles()
        # Backward motion (un-zoom)
        else:
            if zoomed:
                # Restore preview image from stored clean image
                thumb_img = thumb_img_clean.copy()
                # Restore previous circle radius
                circle_radius = round(circle_radius / zoom)
                # Un-set zoomed state
                zoomed = False
                # Repaint circles after un-zooming (refresh image)
                repaint_circles()
    # Store circle (label) coordinates after LMB click
    elif event == cv.EVENT_LBUTTONUP:
        if zoomed:
            # Convert from zoomed preview coordinates to source coordinates
            orig_x, orig_y, orig_r = round(x / zoom * ratio) + shift_x, \
                                     round(y / zoom * ratio) + shift_y, \
                                     round(circle_radius / zoom * ratio)
        else:
            # Convert from preview coordinates to source coordinates
            orig_x, orig_y, orig_r = round(x * ratio), round(y * ratio), round(circle_radius * ratio)
        # Append circle (label) to current image's list of circles (labels)
        circle = (orig_x, orig_y, orig_r, object_class)
        image_circles_dict[list(image_circles_dict.keys())[current_image]].append(circle)
        # Draw circle on screen
        thumb_img = cv.circle(thumb_img, (x, y), circle_radius, object_colors[object_class], 1)


def initialize_images(file_list):
    """ Initializes dict for storing circles (labels) belonging to specific image """
    for file in file_list:
        image_circles_dict[file] = []
    return list(image_circles_dict.keys())


def augment_image(img, r):
    """ Function responsible for generating augmented images from and image.
    The 'img' here is bigger than it is supposed to be, to have room for operations.
    The resulting images are cut with respect to provided 'r' parameter """

    # Find center of image
    h, w = img.shape[:2]
    y, x = round(h / 2), round(w / 2)

    # Prepend an original image
    imgs = [img[y - r:y + r, x - r:x + r], ]

    # Get 3 random angles for image rotation
    for angle in sample(range(0, 180), 3):
        # Rotate with randomized value
        rot = cv.getRotationMatrix2D((x, y), angle, 1)
        rotated = cv.warpAffine(img, rot, (w, h))
        # Crop after rotation
        cropped = rotated[y - r:y + r, x - r:x + r]
        imgs.append(cropped)
        # Get 4 random brightness and contrast values and apply them to images
        for br, con in zip(sample(range(-30, 30), 4), sample(list(np.arange(0.9, 1.2, 0.05)), 4)):
            hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(con * hsv[:, :, 2] + br, 0, 255)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            imgs.append(bgr)
    return imgs


def cut_images(output_dir, remove):
    """ Function that cuts out image fragments and preform augmentation on it """

    # Count of images in output dir, for image naming
    img_count = len(list(Path(output_dir).iterdir()))

    # Iterate over image: circles dict
    for image_file, circle_list in image_circles_dict.items():
        if circle_list:
            # Load the image
            img = cv.imread(str(image_file))
            for x, y, r, _ in circle_list:
                # Adjust boundaries to account for rotations of image
                bigger_r = round(r * np.sqrt(2))
                # Cut image
                img_to_aug = img[y - bigger_r: y + bigger_r, x - bigger_r:x + bigger_r]
                # Augment the cut image
                images = augment_image(img_to_aug, r)
                # Save the images to disk
                for the_img in images:
                    cv.imwrite(f'{output_dir}/bolt_{img_count:06}.png', the_img)
                    img_count += 1

            # Remove original file if parameter was used by user
            if remove:
                image_file.unlink()


def label_images(output_dir, label_format, remove):
    """ Function responsible for generating label *.txt files for different formats """
    if label_format == 'yolov5':
        #  YOLO format, with one *.txt file per image, stored in 'labels' directory next to the 'images' directory:
        #  - One row per object
        #  - Each row is (class x_center y_center width height) format.
        #  - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide
        #  x_center and width by image width, and y_center and height by image height.

        # Prepare directories
        labels_dir = Path(output_dir) / 'labels'
        images_dir = Path(output_dir) / 'images'
        labels_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)

        # Process all images
        for k, v in image_circles_dict.items():
            if v:
                # Create new file paths
                label_file = labels_dir / (k.stem + '.txt')
                image_file = images_dir / k.name

                # Copy from source location to output location
                copyfile(k, image_file)

                # Get image dimensions
                img = cv.imread(str(k))
                img_h, img_w = img.shape[:2]

                # Write label file with normalized format (description in header)
                with label_file.open('w+') as f:
                    for x, y, r, cls in v:
                        x /= img_w
                        y /= img_h
                        w = 2 * r / img_w
                        h = 2 * r / img_h
                        bounds_str = ' '.join(f'{i:.6f}' for i in (x, y, w, h))
                        f.write(f'{cls} {bounds_str}\n')

                # Remove original file if parameter was used by user
                if remove:
                    k.unlink()


def run_label_tool(input_dir, output_dir, mode, label_format, remove):
    """ Main function that creates workspace window and handles key events in main loop """
    global current_image, thumb_img, ratio, circle_radius, zoom, object_class, object_colors

    # Load images from specified path into list
    img_list = initialize_images(Path(input_dir).iterdir())

    if not img_list:
        print('Images directory is empty!')
        return -1

    # Create window and set callback function for mouse handling
    cv.namedWindow('image')
    cv.moveWindow('image', 0, 0)
    cv.setMouseCallback('image', mouse_event)

    # Load first image and make thumbnail from it (resize it to fit screen for better visual sensation)
    thumb_img = load_image(str(img_list[current_image]))

    # Store ratio of conversion from original size to thumbnail size
    ratio = orig_img.shape[0] / window_height

    # Main loop
    while True:
        # Draw circle on cursor position
        temp = thumb_img.copy()
        cv.circle(temp, (mouse_x, mouse_y), circle_radius, object_colors[object_class], 1)
        cv.imshow('image', temp)

        # Wait for action
        k = cv.waitKey(15)

        # Action next photo, D key
        if k == ord('d'):
            # Increment current index and load next image
            current_image = (current_image + 1) % len(img_list)
            thumb_img = load_image(str(img_list[current_image]))
            # Repaint circles after changing image (refresh image)
            repaint_circles()
        # Action prev photo, A key
        if k == ord('a'):
            # Decrement current index and load previous image
            current_image = (current_image - 1) % len(img_list)
            thumb_img = load_image(str(img_list[current_image]))
            # Repaint circles after changing image (refresh image)
            repaint_circles()
        # Action clear last selection, C key
        elif k == ord('c'):
            # If any circles on current image
            if image_circles_dict[img_list[current_image]]:
                # Remove last one
                image_circles_dict[img_list[current_image]].pop()
                # Load clean image
                if zoomed:
                    thumb_img = zoomed_img_clean.copy()
                else:
                    thumb_img = thumb_img_clean.copy()
                # Repaint circles after removing from image (refresh image)
                repaint_circles()
        # Action increase radius, W key
        elif k == ord('w'):
            circle_radius = np.clip(circle_radius + 1, circle_radius, round(window_width / 2))
        # Action decrease radius, S key
        elif k == ord('s'):
            circle_radius = np.clip(circle_radius - 1, 0, circle_radius)

        # Action increase zoom, =/+ key
        elif k == ord('='):
            zoom = np.clip(zoom + 0.5, zoom, 10)
        # Action decrease zoom, -/_ arrow key
        elif k == ord('-'):
            zoom = np.clip(zoom - 0.5, 2, zoom)

        # Action change class to 0 - bolt, '1' key
        elif k == ord('1'):
            object_class = 0
        # Action change class to 1 - lamp, '2' key
        elif k == ord('2'):
            object_class = 1
        # Action cut images or export labels, E key
        elif k == ord('e'):
            if mode == 'cut':
                cut_images(output_dir, remove)
            else:
                label_images(output_dir, label_format, remove)
            break
        # Action exit program, ESC or Q keys
        elif k == 27 or k == ord('q'):
            break

    # Shutdown OpenCV windows
    cv.destroyAllWindows()


def repaint_circles():
    """ Draw all current image's circles with conversion from source coordinates to preview coordinates """
    global thumb_img
    if zoomed:
        # List all image paths
        image_paths = list(image_circles_dict.keys())
        # Get path for the current image
        current_image_path = image_paths[current_image]
        # Loop over all labels (circles) for current image (path is a key, labels are values)
        for x, y, r, c in image_circles_dict[current_image_path]:
            thumb_x, thumb_y, thumb_r = round((x - shift_x) / ratio * zoom), \
                                        round((y - shift_y) / ratio * zoom), \
                                        round(r / ratio * zoom)
            thumb_img = cv.circle(thumb_img, (thumb_x, thumb_y), thumb_r, object_colors[c], 1)
    else:
        for x, y, r, c in image_circles_dict[list(image_circles_dict.keys())[current_image]]:
            thumb_x, thumb_y, thumb_r = round(x / ratio), round(y / ratio), round(r / ratio)
            thumb_img = cv.circle(thumb_img, (thumb_x, thumb_y), thumb_r, object_colors[c], 1)


def load_image(img_path):
    """ Function for opening image file and resizing it to fit the visible window """
    global orig_img, window_width, window_height, thumb_img_clean

    # Load img from file
    pil_image = Image.open(img_path)

    # Convert to OpenCV format for later use
    orig_img = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Resize to fit window with aspect ratio preserved
    pil_image.thumbnail((window_width, window_height))

    # Update window size
    window_width, window_height = pil_image.size

    # Convert resized image to OpenCV format
    thumb_img_clean = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    return thumb_img_clean.copy()


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='path to images to be cut / labeled', type=str)
    parser.add_argument('-o', '--output-dir', help='path where to store cut images / label files', type=str)
    parser.add_argument('-m', '--mode', help='cut / label operation mode',
                        choices=['cut', 'label'], default='cut', type=str)
    parser.add_argument('-f', '--label-format', help='format of labels', choices=['yolov5'], type=str)
    parser.add_argument('-r', '--remove', help='remove source file after cutting / labeling',
                        default=False, action='store_true')
    args = parser.parse_args()

    if not args.input_dir or not Path(args.input_dir).exists() \
            or not args.output_dir or not Path(args.output_dir).exists():
        print('Paths were not specified or do not exist!')
        parser.print_help()
        return

    if args.mode == 'label' and not args.label_format:
        print('Label mode was used but no label format was specified!')
        parser.print_help()
        return

    run_label_tool(args.input_dir, args.output_dir, args.mode, args.label_format, args.remove)


if __name__ == '__main__':
    main()
