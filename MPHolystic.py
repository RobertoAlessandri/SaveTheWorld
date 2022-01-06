import cv2
import mediapipe as mp

import imutils
import numpy as np
import argparse
import os
from pythonosc import udp_client

from joblib import dump,load
from hand_detection_utils import *
from SVM import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic;


if __name__ == "__main__":

# Usage info
  print('USAGE:')
  print('	-Before training generate the images for the the two classes press "a" for class 1 and "b" for class 2:')
  print('		-Press "a" to save class A images')
  print('		-Press "b" to save class B images')
  print('	-Press "t" to start SVM training (if a model has already been saved, it will be loaded)')
  print('	-Press "s" to start sound generation (must be pressed after training)')
  print('	-Press "q" to stop sound and "q" to stop image capture')
  
  # initialize weight for running average
  aWeight = 0.5

  num_frames_train = 0

  # initialize num of frames
  num_frames = 0

  # For webcam input:
  cap = cv2.VideoCapture(0)

  # Initialize variables
  TRAIN = False  # If True, images for the classes are generated
  SVM = False  # If True classification is performed
  START_SOUND = False  # If True OSC communication with SC is started

  with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # flip the frame so that it's not in the mirror view
      image = cv2.flip(image, 1)

      # clone the frame
      clone = image.copy()

      # get high and width of the frame
      (heigth, width) = image.shape[:2]

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

      #image_segment = segment(image)

      #if image_segment is not None:
        #(thresholded, segmented) = image_segment


      # increment the number of frames
      num_frames += 1

      if cv2.waitKey(5) & 0xFF == 27:
        break

      if TRAIN:
    
			# Check if directory for current class exists
        if not os.path.isdir('images/class_'+class_name):
          os.makedirs('images/class_'+class_name)

        if num_frames_train < tot_frames:
				  # Change rectangle color to show that we are saving training images
          text = 'Generating ' + str(class_name) + ' images'
          cv2.putText(clone, text, (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				
				  # Save training images corresponding to the class
          cv2.imwrite('images/class_'+class_name+'/img_'+str(num_frames_train)+'.png', image)

				  # keep track of how many images we are saving
          num_frames_train += 1

        else:

          print('Class '+class_name+' images generated')
          TRAIN = False

      if SVM:
			  # Convert image frame to numpy array
        image_vector = np.array(image)

			  # Use trained SVM to  predict image class
        class_test = model.predict(image_vector.reshape(1, -1))

        if class_test == 0:
				  # print('Class:  A value: ('+str(c_x)+','+str(c_y)+')')
          text = 'Class: A'
        else:
				  # print('Class: B value: ('+str(c_x)+','+str(c_y)+')')
          text = 'Class: B'

        cv2.putText(clone, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			  # Here we send the OSC message corresponding
        if START_SOUND:
          if class_test == 0:
					#freq = (c_x/width_roi) * 100
					#amp = (c_y/height_roi) 
            client.send_message('/globe_control', 1)
					#print(client.send_message('/globe_control'))
          else:
					#detune = (c_x/width_roi) * 0.1
					#lfo = (c_y/height_roi) * 10
            client.send_message('/globe_control', 0)
					#print(client.send_message('/globe_control'))

      # observe the keypress by the user
      keypress = cv2.waitKey(1) & 0xFF

		
		  # if the user pressed "q", then stop looping
      if keypress == ord("q"):
        break

		  # Generate class A images
      if keypress == ord("a"):
        print('Generating the images for class A:')
        TRAIN = True
        num_frames_train = 0
        tot_frames = 250
        class_name = 'a'

		  # Generate class B images
      if keypress == ord("b"):
        print('Generating the images for class B:')
        TRAIN = True
        num_frames_train = 0
        tot_frames = 250
        class_name = 'b'

		  # Train and/or start SVM classification
      if keypress == ord('t'):
        SVM = True

        if not os.path.isfile('modelSVM.joblib'):
          model = train_svm()
        else:
          model = load('modelSVM.joblib')

		  # Start OSC communication and sound
      if keypress == ord('s'):
        START_SOUND = True

			  # argparse helps writing user-friendly commandline interfaces
        parser = argparse.ArgumentParser()
			  # OSC server ip
        parser.add_argument("--ip", default='127.0.0.1', help="The ip of the OSC server")
			  # OSC server port (check on SuperCollider)
        parser.add_argument("--port", type=int, default=57120, help="The port the OSC server is listening on")

			  # Parse the arguments
        args = parser.parse_args()

			  # Start the UDP Client
        client = udp_client.SimpleUDPClient(args.ip, args.port)

		  # Stop OSC communication and sound
      if keypress == ord('q'):

			  # Send OSC message to stop the synth
        client.send_message("/globe_control", ['stop'])
  

  cap.release()
  cv2.destroyAllWindows()