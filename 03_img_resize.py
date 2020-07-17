import os
import pathlib
import glob
import cv2
import numpy as np

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 경로에서 이미지 파일 취득
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대 경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지 파일(파일명):{filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None:
            print(f"이미지 파일[{fullpath}]을 읽을 수가 없습니다.")
            continue
        name_images.append((filename, image))
    return name_images


def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1

# Face Image Directory
IMAGE_PATH_PATTERN = "./face_image_before/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_image"

def main():
    print("===================================================================")
    print("이미지 resize")
    print("===================================================================")

    # 디렉토리 작
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)


    # 대상 이미지 읽기
    name_images = load_name_images(IMAGE_PATH_PATTERN)

    # 대상 이미지 별로 증가 작업
    for name_image in name_images:
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]# image 값
        image = cv2.resize(image,(64,64))
        output_path = os.path.join(OUTPUT_IMAGE_DIR,f"{filename}{extension}")
        print(f"출력 파일(절대 경로) : {output_path}")
        cv2.imwrite(output_path, image)            

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()