import numpy as np
import cv2
import os
import glob
import platform
import re


def get_host_platform():
    """
    get the  host system, which could be used for platform checks

    :return: tuple with  os_name, sys, sys_release
    """
    os_name = os.name()
    sys = platform.system()
    sys_release = platform.release()
    return os_name, sys, sys_release


def get_all_images_from_part(read_path, part_name, extension):
    """
    generates a list of all images in the given read_path with the given extension

    :param read_path: string,  path of the data dir
    :param part_name: sting, name of the part
    :param extension: string, extension like jpg
    :return: list of all part paths
    """
    numbers = filter(None, [re.match('\d+', f) for f in os.listdir(read_path)])
    pats_path = [x.string for x in numbers if x is not None]

    if pats_path is []:
        print("ERROR: no director's in path or wrong naming")
        SystemExit

    if part_name not in pats_path:
        if type(part_name) is str:
            print("ERROR: dir for this part number doesn't exists")
            SystemExit
        else:
            part_name = str(part_name)

            if part_name not in pats_path:
                print("ERROR: dir for this part number doesn't exists")
                SystemExit

    if extension not in ['gif', 'jpeg', 'jpg', 'png']:
        print('ERROR: unknown extension')
        SystemExit

    else:
        data_path = os.path.join(os.path.join(read_path, part_name), '*{}'.format(extension))
        files = glob.glob(data_path)
        return files


def get_part_from_image(image, scale=1.3):

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours

    filtered_contours = []
    for c in cnt:
        if len(c) > 50:
            filtered_contours.append(c)
    if len(filtered_contours) > 1:
        merged_contours = np.concatenate(filtered_contours)
        (x, y), radius = cv2.minEnclosingCircle(merged_contours)
        center = (int(x), int(y))
        radius = int(radius)
        scaled_rad = int(radius * scale)
        img_cutted = image[center[1] - scaled_rad: center[1] + scaled_rad,
                           center[0] - scaled_rad: center[0] + scaled_rad,
                           :]

        return img_cutted
    else:
        return None


def resize_img(image, min_dimensions=100, resize=False, new_size=100, set_gray=False):

    if resize is True and new_size > min_dimensions:
        print('ERROR: new_size has to be smaller than min_dimensions.')
    else:

        if image is None:
            print('Image {} does not exist.'.format(img_name))

        if set_gray is True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape
        else:
            height, width, c_channel = image.shape

        if resize is True:  # Anpassen auf neue größe, wenn resize == True
            if min(height, width) < new_size:  # prüfe ob Bild bereits kleiner als neue größe (new_size) ist
                print('Image {} does not have the necessary dimensions.'.format(img_name))
            else:
                if height < width:
                    cut = 0.5 * (width - height)  # berechne wieviel pro Seite abzuschneiden ist
                    cut = int(cut)
                    cutted_img = image[0:height, cut:(width - cut)]  # Breite des Bildes wird angepasst -> quadratisch
                elif height > width:  # umgekehrter falls, sonst wie eben
                    cut = 0.5 * (height - width)
                    cut = int(cut)
                    cutted_img = image[cut:(height - cut), 0:width - cut]
                else:
                    cutted_img = image

                resize_image = cv2.resize(cutted_img,
                                          (new_size, new_size))  # quadratische Bild wird auf neue angepasst
                return resize_image

        else:
            if min(height, width) < min_dimensions:
                print('Image {} does not have the necessary dimensions.'.format(img_name))
            else:
                return None


print(get_host_platform)

part_nr = '2'
extension = 'JPG'
read_path = '/home/jeronimo/git/AUT-CNN-TUB/Data/Datensatz/'
write_path = '/home/jeronimo/git/AUT-CNN-TUB/Data/Centered/'
new_size = 100
set_gray = False

files = get_all_images_from_part(read_path, part_nr, extension)
for i, file_path in enumerate(files):
    img_name = file_path.split('/')[-1]
    image = cv2.imread(file_path)
    try:
        img_centered = get_part_from_image(image)
        if img_centered is not None:
            img_resized = resize_img(img_centered, resize=True, new_size=new_size, set_gray=set_gray)

            part_patch = os.path.join(write_path, part_nr)
            if img_resized is not None:
                if os.path.exists(part_patch):
                    cv2.imwrite(os.path.join(part_patch, '{}.{}'.format(img_name, extension)),
                                img_resized)
                else:
                    os.makedirs(part_patch)
                    cv2.imwrite(os.path.join(part_patch, '{}.{}'.format(img_name, extension)),
                                img_resized)
    except:
        print('ERROR:{} couled nor be procesd'.format(img_name))
