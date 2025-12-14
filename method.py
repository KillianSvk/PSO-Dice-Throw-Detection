from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_DIR = Path('images')
MIN_AREA_THRESHOLD = 500 

def load_image(path):
    img = cv2.imread(str(path))
    value = int(path.name[0])
    
    if value < 1 or value > 6:
        raise Exception(f"Error while loading {path}")
    
    return (img, value)


def main():
    count = 0
    img_paths = IMG_DIR.glob('*.png')
    all_count = len(list(img_paths))

    for path in IMG_DIR.glob('*.png'):
        img, value = load_image(path)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
        
        thresh_img = cv2.adaptiveThreshold(
            blur_img,
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            3,
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh_img = cv2.morphologyEx(
            thresh_img, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=1
        )

        kernel = np.ones((5, 5), np.uint8)
        thresh_img = cv2.morphologyEx(
            thresh_img, 
            cv2.MORPH_DILATE, 
            kernel, 
            iterations=2
        )

        contours, hierarchy = cv2.findContours(
            thresh_img.copy(), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
                
        dice_contour = None
        max_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            
            if area > MIN_AREA_THRESHOLD and area > max_area:
                max_area = area
                dice_contour = c
        
        if dice_contour is not None:
            contour_img = img.copy()
            cv2.drawContours(contour_img, [dice_contour], -1, (0, 255, 0), 2) 
            
            mask = np.zeros(blur_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [dice_contour], -1, 255, cv2.FILLED) 
    
            isolated_dice = cv2.bitwise_and(blur_img, blur_img, mask=mask)

            isolated_dice_binary = cv2.adaptiveThreshold(
                isolated_dice, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                3
            )

            kernel = np.ones((3, 3), np.uint8)
            isolated_dice_binary = cv2.morphologyEx(
                isolated_dice_binary, 
                cv2.MORPH_OPEN, 
                kernel, 
                iterations=1
            )
           
            contours, hierarchy = cv2.findContours(
                isolated_dice_binary,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE
            )

            MIN_DOT_AREA = 30
            MAX_DOT_AREA = 100
            MIN_ROUNDNESS = 0.5

            dots_final = img.copy()
            dots_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, closed=True)

                if area < MIN_DOT_AREA or area > MAX_DOT_AREA:
                    continue
                
                roundness = (4 * np.pi * area) / (perimeter ** 2)

                if roundness < MIN_ROUNDNESS:
                    continue

                dots_count += 1
                cv2.drawContours(dots_final, [contour], -1, (0, 255, 0), 1)

            print(f"Detected {dots_count} / {value}")

        if dots_count == value:
            count += 1

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        images = [
            (img[:,:,::-1], "Original"),
            (gray_img, "1. Grayscale"),
            (blur_img, "2. Gaussian Blur"),
            (thresh_img, "3. Binary + Opening"),
            (contour_img, "4. Contour Found"),
            (dots_final[:, :, ::-1], "Final")
        ]
        
        for i, (image, title) in enumerate(images):
            ax = axes[i]
            
            if image.ndim == 2:
                ax.imshow(image, cmap='gray') 
            else:
                ax.imshow(image)
                
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
        plt.tight_layout()

        plt.show(block=True) 


    print(f"{count}/{all_count}")
            

if __name__ == "__main__":
    main()