import numpy as np
import cv2
from sklearn import cluster

def get_blobs(img):
   params = cv2.SimpleBlobDetector_Params()
   params.filterByInertia = True
   params.minInertiaRatio = 0.6
   detector = cv2.SimpleBlobDetector_create(params)

   # blurred_img = cv2.GaussianBlur(img, 5)
   gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
   blobs = detector.detect(gray_img)
   return blobs


def get_dice_from_blobs(blobs):
   # Get centroids of all blobs
   X = []
   for b in blobs:
      pos = b.pt

      if pos != None:
         X.append(pos)

   X = np.asarray(X)

   if len(X) > 0:
      # Important to set min_sample to 0, as a dice may only have one dot
      clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)

      # Find the largest label assigned + 1, that's the number of dice found
      num_dice = max(clustering.labels_) + 1

      dice = []

      # Calculate centroid of each dice, the average between all a dice's dots
      for i in range(num_dice):
         X_dice = X[clustering.labels_ == i]

         centroid_dice = np.mean(X_dice, axis=0)

         dice.append([len(X_dice), *centroid_dice])

      return dice

   else:
      return []
   
def overlay_info(frame, dice, blobs):
   # Overlay blobs
   for b in blobs:
      pos = b.pt
      r = b.size / 2

      cv2.circle(frame, (int(pos[0]), int(pos[1])),
                  int(r), (255, 0, 0), 2)

   # Overlay dice number
   for d in dice:
      # Get textsize for text centering
      textsize = cv2.getTextSize(
         str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

      cv2.putText(frame, str(d[0]),
                  (int(d[1] - textsize[0] / 2),
                  int(d[2] + textsize[1] / 2)),
                  cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

if __name__ == "__main__":
   img = cv2.imread("images/5-3.png")

   # We'll define these later
   blobs = get_blobs(img)
   dice = get_dice_from_blobs(blobs)
   out_frame = overlay_info(img, dice, blobs)

   cv2.imshow("frame", img)

   res = cv2.waitKey(0)