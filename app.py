from flask import Flask
app = Flask(__name__)

@app.route('/')
def homepage():
    return """
    <iframe src="https://share.streamlit.io/agung5114/azurecledo/main/app.py" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>
    """

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)