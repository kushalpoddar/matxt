from flask import Flask, jsonify
from flask_restful import Api, Resource, abort, reqparse, request
from flask_cors import CORS
import json
import werkzeug
from hashlib import md5
from time import localtime
import requests
from cv2 import cv2
from tensorflow.keras.models import model_from_json
from pytexit import py2tex
#Predicting
model_json = open('model/model.json', 'r')
loaded_model_json = model_json.read()
model_json.close()
loaded_model = model_from_json(loaded_model_json)

print('Loading weights...')
loaded_model.load_weights("model/model_weights.h5")

app = Flask(__name__, static_folder="temp_save")
CORS(app)
api = Api(app)



def sortRelativeAsa(x):
	return x['rel_asa']

class Asaview(Resource):
	def post(self):
		#Handling the file upload of PDB
		parse = reqparse.RequestParser()
		parse.add_argument('fileimg', type=werkzeug.datastructures.FileStorage, location='files')
		
		args = parse.parse_args()

		image_file = args['fileimg']
		random_hash = md5(str(localtime()).encode('utf-8')).hexdigest()
		filepath = "temp/" + random_hash + ".png"
		image_file.save(filepath)

		image = cv2.imread(filepath);
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(filepath, gray)

		image = cv2.imread(filepath);
		rect, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
		ctrs, hier = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


		rects=[]
		for c in cnt :
		    x,y,w,h= cv2.boundingRect(c)
		    rect=[x,y,w,h]
		    rects.append(rect)

		i=0
		imgurls = []

		text_str = ''
		for r in rects:
			x=r[0]
			y=r[1]
			w=r[2]
			h=r[3]
			im_crop =binary[y:y+h+5,x:x+w+5]

			# img_not = cv2.bitwise_not(im_resize)

			#Size of the image
			height, width = im_crop.shape
			img_area = height*width

			val_type = 'char'
			#Checking if the area of this image is half the area of previous image
			if len(imgurls) and img_area < imgurls[i-1]['area']/1.5 and (y+h) < 50:
				val_type = 'power'

			img_not = cv2.resize(im_crop, (28, 28),  
			       interpolation = cv2.INTER_NEAREST)

			
			img_not = cv2.copyMakeBorder(img_not.copy(),5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,0])

			img_not = cv2.resize(img_not, (28, 28),  
			       interpolation = cv2.INTER_NEAREST)
			
			# cv2.imshow("Invert1",result)
			# cv2.waitKey(0)

			# im_resize = cv2.resize(im_crop,(28,28))
			filename_small = 'temp_save/'+random_hash + '_' + str(i)+'.png'
			cv2.imwrite(filename_small, img_not)


			img_model = img_not.reshape(-1, 28, 28, 1)
			result = loaded_model.predict_classes(img_model)

			prediction = int(result[0])


			if prediction == 10:
				prediction = '+'
			elif prediction == 11:
				prediction = '-'
			elif prediction == 12:
				prediction = '*'
			else:
				prediction = str(prediction)

			if val_type == 'char':
				text_str = text_str + prediction
			else:
				text_str = text_str + '**' +prediction

			imgurls.append({ 'name' : filename_small, 'prediction' : prediction, 'area' : img_area, 'type' : val_type })

			i = i+1

		evaluation = eval(text_str)
		latex = py2tex(text_str)
		latex = latex[2:len(latex)-2]
		
		return jsonify({ 'text' : text_str, 'eval' : evaluation, 'latex' : latex, 'pred' : imgurls })

api.add_resource(Asaview, '/list')

if __name__ == '__main__':
	app.run(debug=True)
