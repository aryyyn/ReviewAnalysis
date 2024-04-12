from flask import Flask,request,jsonify,render_template_string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)


review_form = '''
<!DOCTYPE html>
<html>
<head>
    <title>Review Analysis</title>
</head>
<body>
    <h1>Submit a Review for Analysis</h1>
    <form action="/review-analysis" method="post">
        <textarea name="review_text" rows="5" cols="40" placeholder="Enter review text here..."></textarea><br><br>
        <input type="submit" value="Submit Review">
    </form>
</body>
</html>
'''



@app.route('/')
def hello():
    return render_template_string(review_form)


@app.route('/review-analysis', methods=['POST'])
def predict_review():
    review_text = request.form.get('review_text')
    
    cv = CountVectorizer()
    loaded_clf = joblib.load('../restaurant_review_model.pkl')
    X_new = cv.transform([review_text]).toarray()
    predictions = loaded_clf.predict(review_text)
    print("Predictions:", predictions)

    return '', 204
    

if __name__ == '__main__':
    app.run()



