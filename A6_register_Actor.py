import cv2
import numpy as np
import time
import ray
ray.init(num_cpus=10)

@ray.remote
class Register_Task:
    def __init__(self):
        # Create ORB detector with 5000 features.
        self.orb_detector = cv2.ORB_create(5000)
        #print('register actor created \n')
        
    def register(self,img1_color,img2_color):
        '''Registe two images'''
        start6 = time.time()
        # Convert to grayscale.
        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        height, width = img2.shape
        #print('Time cvt color:',time.time()-start2)
        
        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        #  (which is not reqiured in this case).
        kp1, d1 = self.orb_detector.detectAndCompute(img1, None)
        kp2, d2 = self.orb_detector.detectAndCompute(img2, None)
          
        # Match features between the two images.
        # We create a Brute Force matcher with 
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
          
        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)
          
        # Sort matches on the basis of their Hamming distance.
        matches.sort(key = lambda x: x.distance)
          
        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches)*90)]
        no_of_matches = len(matches)
          
        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
          
        for i in range(len(matches)):
          p1[i, :] = kp1[matches[i].queryIdx].pt
          p2[i, :] = kp2[matches[i].trainIdx].pt
          
        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        #print('Time register:',time.time()-start6)
         
        #start7 = time.time()
        # Use this matrix to transform the
        # colored image wrt the reference image.
        transformed_img = cv2.warpPerspective(img1_color,
                            homography, (width, height))
        print('Time register and warp:',time.time()-start6)
        
        return transformed_img
        
if __name__=='__main__':
    
    for j in range(5):
        n = j+1 #actor count
        reg_objs  = [Register_Task.remote() for _ in range(n)]
        
        start0 = time.time()
        # Read images
        start1 = time.time()
        img1_colora = cv2.imread("moving.jpg")  # Image to be aligned.
        img2_colora = cv2.imread("ref.jpg")    # Reference image.
        print('Time read:',time.time()-start1,'\n')
        
        
        resustl_refs = []
        for i in range(n):
            # remote put images
            start2 = time.time()
            img1_color = ray.put(img1_colora)
            img2_color = ray.put(img2_colora)
            print('Time ray put:',time.time()-start2)
            
            # remote register id
            start3 = time.time()
            transformed_img_ref = reg_objs[i].register.remote(img1_color,img2_color)
            resustl_refs.append(transformed_img_ref)
            print('Time ref id:',time.time()-start3)
            
        # ray get resutls from id   
        start4 = time.time()
        transformed_imgs = ray.get(resustl_refs)
        print('Ray register get:',time.time()-start4)
        
        
        for i in range(n):
            start5 = time.time()
            transformed_img = transformed_imgs[i]
            cv2.imwrite('output.jpg', transformed_img)
            print('Time write:',time.time()-start5,'\n')
            
        print('## Time per reg:',(time.time()-start0)/n) 
            
        
        
