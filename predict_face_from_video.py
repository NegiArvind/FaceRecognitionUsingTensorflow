# -*- coding: utf-8 -*-

# This script is used to predict the person in an image 

import cv2
import dlib
import time
import tensorflow as tf
import pickle
import os
import numpy as np
from tensorflow.python.platform import gfile
from matplotlib import pyplot as plt
from align_dlib import AlignDlib
import sys

# for drive
#driveLocation='drive/My Drive/FaceRecognition/'
# for pc
driveLocation=''
predictor_model=driveLocation+"shape_predictor_68_face_landmarks.dat"


def main(videoLocation, model_path, classifier_output_path):
    """
    Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
     a classifier outputted to :param output_path
     
    :param input_directory: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param classifier_output_path: Path to write pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_labels: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    :param is_train: bool denoting if training or evaluate
    """

#     img = cv2.imread(imageLocation)
#     img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.rcParams["axes.grid"] = False #remove whitelines from the image
#     plt.imshow(img_cvt)
#     plt.show()
    
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        
        video_capture=cv2.VideoCapture(videoLocation)
        while video_capture.isOpened():
            ret,frame=video_capture.read()
            
            index = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if (index % 60 == 0):
                print(index)
                #img_cvt=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.rcParams["axes.grid"] = False #remove whitelines from the image
                plt.imshow(img_cvt)
                plt.show()
                aligned_images,boundary_boxes=detect_and_align_faces(frame) # contains list of aligned face
                imgs=[]
                for image in aligned_images:
                    #print("image size,"+str(image.shape))
                    image=prewhiten(image) # it whitens the image
                    resize_image=cv2.resize(image,(160,160),interpolation=cv2.INTER_CUBIC) # reshape image 3d matrix(234,234,3)
                    # into (160,160,3)
                    resize_image=resize_image.reshape(-1,160,160,3)
                    imgs.append(resize_image)
                    #print("image after resize,"+str(resize_image.shape))


        #         print(imgs)

                _load_model(model_filepath=model_path)

                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                emb_array =_create_embeddings(embedding_layer, imgs,images_placeholder,
                                                            phase_train_placeholder, sess)

                print('Created {} embeddings'.format(len(emb_array)))

                classifier_filename = classifier_output_path
                recognize_face(emb_array,classifier_filename,frame,boundary_boxes)
                print('Completed in {} seconds'.format(time.time() - start_time))
                break;


def _create_embeddings(embedding_layer, images,images_placeholder, phase_train_placeholder, sess):
    
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer: 
    :param images: - images for which you have to create embedding(128 decimal output)
    :param labels: -labels of images
    :param images_placeholder: 
    :param phase_train_placeholder: 
    :param sess: 
    :return: (tuple): image embeddings and labels
    """
    print("Start creating embedding")
    emb_array = None
    try:
      for i in range(len(images)):
          emb = sess.run(embedding_layer,
                             feed_dict={images_placeholder: images[i], phase_train_placeholder: False})

#           print("emb")
#           print(emb)
          emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb

#     try:
#         i = 0
#         while True:
#             emb = sess.run(embedding_layer,
#                            feed_dict={images_placeholder: images, phase_train_placeholder: False})

#             print("emb")
#             print(emb)
#             emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
#             i += 1
    except tf.errors.OutOfRangeError:
        pass

    
#     print(emb_array)
    return emb_array

def detect_and_align_faces(img):
    face_aligner=AlignDlib(predictor_model)
    face_detector=dlib.get_frontal_face_detector()
    
    np.uint8(img)
    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(img, 1)
    
    print("I found {} faces in the ".format(len(detected_faces)))
        
    aligned_faces=[]
    boundary_boxes=[]
    #Loop through each face we found in our image
    for i,face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".
              format(i+1, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        # Aligning the face in correct position.
        alignedFace=face_aligner.align(imgDim=234,rgbImg=img,bb=face_rect,
                                       landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        aligned_faces.append(alignedFace)
        boundary_boxes.append(face_rect)
    
    return aligned_faces,boundary_boxes

def recognize_face(emb_array,classifier_filename,image,boundary_boxes):
    
    if not os.path.exists(classifier_filename):
        raise ValueError('Pickled classifier not found, have you trained first?')
    
    with open(classifier_filename,'rb') as f:
        
        model,class_names=pickle.load(f)
        print("class_names")
        print(class_names)
        predictions=model.predict_proba(emb_array,)
        print("predictions")
        print(predictions)
        best_class_indices = np.argmax(predictions, axis=1)
        
        print("best_class_indices")
        print(best_class_indices)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        
        print("Prediction Result: ")
        person_names=[]
        for i in range(len(best_class_indices)):
            person_name=''
            if best_class_probabilities[i]>0.52:
                 person_name=class_names[best_class_indices[i]]
            else:
                 person_name='Unknown'
            person_names.append(person_name)
            print('%4d  %s: %.3f' % (i,person_name, best_class_probabilities[i]))
        showPhoto(person_names,best_class_probabilities,image,boundary_boxes)    
        

def showPhoto(person_names,best_class_probabilities,image,boundary_boxes):
    image=np.uint8(image)
    for i in range(len(best_class_probabilities)):
         face_rect=boundary_boxes[i]
         person_name=person_names[i]
         image=cv2.rectangle(image,(face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()),
                  (0,255,0),2) #(0,255,0) is the combination of color
         text_x=face_rect.left()+5
         text_y=face_rect.top()-20
         image=cv2.putText(image,person_name+" - "+str('%.2f'%(best_class_probabilities[i]*100))+"%",(text_x, text_y), cv2.FONT_HERSHEY_COMPLEX,
                0.6, (0, 255,0), thickness=1, lineType=cv2.LINE_AA)
#     cv2.imshow('image',image)
#     cv2.waitkey(0)
#    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    plt.rcParams["axes.grid"] = False #remove whitelines from the image
#    plt.imshow(image)
#    plt.show()
    cv2.imshow("After Recognition process",image)
    key_pressesd=cv2.waitKey(0)
    if key_pressed==13 or key_pressed==32: #unicode for enter key and space key
        cv2.destroyAllWindows()
        
    
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
  
def _load_model(model_filepath):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Missing model file. Exiting')
        sys.exit(-1)
    print("Model loaded successfully")


if __name__=='__main__':
    
    main(videoLocation=driveLocation+'video/video1.mp4',model_path=driveLocation+'20180402-114759_model/20180402-114759.pb',classifier_output_path=driveLocation+'classifier.pkl')
