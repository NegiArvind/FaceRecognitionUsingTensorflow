import dlib
import cv2
from align_dlib import AlignDlib # This is python file saved in this working directory
import os

imageLocation='dataset/input'
imageOutputLocation="dataset/AlignedOutput"
# we can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model="shape_predictor_68_face_landmarks.dat"
face_aligner=None
face_detector=None

# Either we can use below method to load image or io.imread(imageLocation) to read method

# img=image.load_img(imageLocation)
# img=np.uint8(img) # it will convert Image type into uint8 type


#img = io.imread(imageLocation)
def preprocess(imageLocation,imageOutputLocation):
    
    if not os.path.exists(imageOutputLocation):
        os.makedirs(imageOutputLocation)
    for input_dir in os.listdir(imageLocation):
        image_output_dir=os.path.join(imageOutputLocation,input_dir)
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
        image_input_class_dir=os.path.join(imageLocation,input_dir)
        for i,image_name in enumerate(os.listdir(image_input_class_dir)):
            newImageName=input_dir+str(i+1)+".jpg"
            os.rename(image_input_class_dir+"/"+image_name,image_input_class_dir+"/"+newImageName)
            detect_and_align_faces(os.path.join(image_input_class_dir,newImageName),os.path.join(image_output_dir,newImageName))
    

def initialize():
    global face_aligner
    global face_detector
    # Create a HOG Face detector using built indlib class
    face_detector=dlib.get_frontal_face_detector()
    #  To deal with the problem that faces turned different directions look totally different to a computer:
    # we use face_pose_predictor package to deal with it
    # We uses face_landmark estimation algorithm (68 specific points)
    # face_pose_predictor=dlib.shape_predictor(predictor_model)
    # To align the image we need to make an object of AlignDlib class
    face_aligner=AlignDlib(predictor_model)

    

# win=dlib.image_window() # it will open a new window to show the image

def detect_and_align_faces(imageInputLocation,imageOutputLocation):
    
    print("detect "+imageInputLocation+" "+imageOutputLocation)
    img = cv2.imread(imageInputLocation)
    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(img, 1)
    
    print("I found {} faces in the file {}".format(len(detected_faces), imageInputLocation))
    
    # Open a window on the desktop showing the image
    #win.set_image(img)
    
    #Loop through each face we found in our image
    for i,face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".
              format(i+1, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        # Draw a box around each face we found
        # win.add_overlay(face_rect)
        
        # Get the the face's pose
        # pose_landmarks = face_pose_predictor(img, face_rect)
        # Aligning the face in correct position.
        alignedFace=face_aligner.align(imgDim=234,rgbImg=img,bb=face_rect,
                                       landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        # imgDim = The edge length in pixels of the square the image is resized to
        
        #Draw the face landmarks on the image
        # win.add_overlay(pose_landmarks)
        
        # Save the aligned image to a file
        cv2.imwrite(imageOutputLocation, alignedFace)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()

if __name__=="__main__":
    initialize()
    preprocess(imageLocation,imageOutputLocation)
