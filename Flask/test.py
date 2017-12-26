from flask import Flask, url_for
app = Flask(__name__)
@app.route('/')
def hello_world():
    return "Hello World"

@app.route('/home')
def index():
    return "This is home page!"

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return 'Post %d' % post_id

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

@app.route('/test')
def test():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='nbgao'))

# HTTP method
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        do_the_login()
    else:
        show_the_login_form()
    

if __name__=='__main__':
    app.run()
