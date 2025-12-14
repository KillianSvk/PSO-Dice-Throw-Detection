import numpy as np
import cv2
from pathlib import Path
from sklearn import cluster

def get_blobs(img):
   params = cv2.SimpleBlobDetector_Params()

   params.filterByArea = True
   params.maxArea = 500

   params.filterByCircularity = True
   params.minCircularity = 0.8

   params.filterByInertia = True
   params.minInertiaRatio = 0.8

   detector = cv2.SimpleBlobDetector_create(params)

   blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
   gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
   
   keypoints = detector.detect(gray_img)
   return keypoints
      
def get_ground_truth(path):
   path = str(path)
   filename = path.split("\\")[-1]
   return filename[0]

if __name__ == "__main__":
   IMG_DIR = Path('images')
   RESULTS_DIR = Path('results')
   RESULTS_DIR.mkdir(exist_ok=True)

   total = 0
   correct = 0
   for path in (IMG_DIR.glob("*.png")):
      img = cv2.imread(path)

      keypoints = get_blobs(img)

      total += 1
      ground_truth = get_ground_truth(path)
      if len(keypoints) == int(ground_truth):
         correct += 1
      
      output = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

      # result_path = RESULTS_DIR / path.name
      # cv2.imwrite(str(result_path), output)

      cv2.imshow(ground_truth, output)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

   # default 17/26 = 65.38%
   # maxarea 17/26 = 65.38%
   # maxarea + mincircularity 22/26 = 84.62%
   # maxarea + mincircularity + mininertiaratio 26/26 = 100.00%
