import face_recognition 
import os
import cv2
import pickle
import time

Known_Faces = 'known_faces'
#Unknown_Faces = 'unknown_faces'
Tolerance = 0.6
Frame_Thickness = 3
Font_Thickness = 2
Model = 'hog'

video = cv2.VideoCapture("1.mp4")

print('Loading Known Faces')

known_faces = []
known_names = []

for name in os.listdir(Known_Faces):
	for filename in os.listdir(f"{Known_Faces}/{name}"):
		#image = face_recognition.load_image_file(f"{Known_Faces}/{name}/{filename}")
		#encoding = face_recognition.face_encodings(image)[0]
		encoding = pickle.load(open(f"{name}/{filename}", "rb"))
		known_faces.append(encoding)
		known_names.append(int(name))

if len(known_names) > 0:
	next_id = max(known_names) + 1
else:
	next_id = 0

print("Processing Unknown Faces")

while True:

	#print(filename)
	#image = face_recognition.load_image_file(f"{Unknown_Faces}/{filename}")
	ret,image = video.read()
	locations = face_recognition.face_locations(image, model = Model)
	encodings = face_recognition.face_encodings(image, locations)
	#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encoding , face_location in zip (encodings,locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
		match = None
		if True in results:
			match = known_names[results.index(True)] 
			print(f"Match Found: {match}")

		else:
			match = str(next_id)
			next_id = next_id + 1
			known_names.append(match)
			known_faces.append(face_encoding)
			os.mkdir(f"{Known_Faces}/{match}")
			pickle.dump(face_encoding, open(f"{Known_Faces}/{match}/{match}-{int(time.time())}.pkl", "wb"))

		top_left = (face_location[3], face_location[0])
		bottom_right = (face_location[1], face_location[2])

		color = [255, 0 , 0]

		cv2.rectangle(image, top_left, bottom_right, color, Frame_Thickness)

		top_left = (face_location[3], face_location[2])
		bottom_right = (face_location[1], face_location[2]+22)

		cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
		cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), Font_Thickness)
	
	cv2.imshow("", image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	#cv2.destroyWindow(filename)		


