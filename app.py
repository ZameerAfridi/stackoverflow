from flask import Flask , render_template, request

app= Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")
@app.route("/sub", methods=["POST"])
def submit():
    if request.method=="POST":
        body=request.form("enterbody")

    return render_template("sub.html", b= body)


if __name__== "__main__":
    app.run(debug=True)