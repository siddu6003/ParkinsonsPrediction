from flask import Flask,render_template,request
from skimage.transform import resize
from skimage.io import imread
import pickle

Categories=['healthy','parkinson']

with open('model_spiral','rb') as f:
    models1=pickle.load(f)

with open('model_random_spiral','rb') as f:
    models2=pickle.load(f)

'''with open('model_KNN_spiral','rb') as f:
    models3=pickle.load(f)'''

with open('model_wave','rb') as f:
    modelw1=pickle.load(f)

with open('model_random_wave','rb') as f:
    modelw2=pickle.load(f)

'''with open('model_KNN_wave','rb') as f:
    modelw3=pickle.load(f)'''

def predict(img_path,mode):
    img=imread(img_path)
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    res=0
    if mode=='svm-spiral':
        res=Categories[models1.predict(l)[0]]
    elif mode=='svm-wave':
        res=Categories[modelw1.predict(l)[0]]
    elif mode=='rf-spiral':
        res=Categories[models2.predict(l)[0]]
    elif mode=='rf-wave':
        res=Categories[modelw2.predict(l)[0]]
    '''
    elif mode=='knn-spiral':
        res=Categories[models3.predict(l)[0]]
    elif mode=='knn-wave':
        res=Categories[modelw3.predict(l)[0]]
    '''
    return res

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def main():
    return render_template("home.html")

@app.route("/upload",methods=['GET','POST'])
def upload():
    return render_template("index.html",name=request.args.get('test'))

@app.route("/submit",methods=['GET','POST'])
def submit():
    f=request.files['file']
    img_path="static/"+f.filename
    f.save(img_path)
    if request.method=='POST' and request.form['test']=='svm-spiral':
        result=predict(img_path,mode='svm-spiral')
    elif request.form['test']=='rf-spiral':
        result=predict(img_path,mode='rf-spiral')
    elif request.form['test']=='svm-wave':
        result=predict(img_path,mode='svm-wave')
    elif request.form['test']=='rf-wave':
        result=predict(img_path,mode='rf-wave')

    return render_template("index.html",prediction=result,img_path=img_path,model=request.form['test'])

if __name__ == "__main__":
    app.run(debug=True)
