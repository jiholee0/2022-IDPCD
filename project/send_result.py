# import sys
# sys.path.append("/main")
# import main as main
# print(main.result)

from flask import Flask, render_template
# from vsearch import search4letters

# Flask 인스턴스 생성
app = Flask(__name__)

@app.route('/init')
def init() ->'html':
    return render_template('./image/screen.html')

app.run()
