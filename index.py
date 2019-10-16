from flask import Flask, render_template, flash, request
import consultaimage as query 
from wtforms import TextField, TextAreaField, StringField, SubmitField, validators
from flask_wtf import Form, FlaskForm
from wtforms.validators import InputRequired
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os, sys


"""
absFilePath = os.path.abspath(__file__)
print(absFilePath)

fileDir = os.path.dirname(os.path.abspath(__file__))
print(fileDir)

fileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(fileDir)
"""


app = Flask(__name__)
app.config['SECRET_KEY'] = 'DontTellAnyone'

photos = UploadSet('photos',IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app,photos)



class formulariosimple(Form):
    nombre = StringField('nombre', validators=[InputRequired()])
    apellido = StringField('apellido', validators=[InputRequired()])

@app.route("/home2",methods = ['GET', 'POST'])
def home():
    form = formulariosimple()
    #print(form.apellido.data)
    if form.validate_on_submit():
        nombrearchivo = query.find_similar_images("Dataset/Bonsai.jpg")
        if len(nombrearchivo) > 0:            
            return render_template('resultado.html', image_name =nombrearchivo )
        else:
            return "La búsqueda no arrojo resultados."

    return render_template('home.html', form=form)

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        nombrearchivo = query.find_similar_images("static/img/"+filename)
        if len(nombrearchivo) > 0:            
            return render_template('resultado.html', image_name =nombrearchivo )
        else:
            return "La búsqueda no arrojo resultados."
        
    return render_template('index.html')

    

#debug=True
"""
if __name__ == '__main__':
    app.run()
"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

