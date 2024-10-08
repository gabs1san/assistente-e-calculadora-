from flask import Flask, jsonify, request
from assistente import detect_facial_expression
from calculadora import count_fingers
from flask import Flask, jsonify, request, render_template 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return "Página de teste funcionando!"



@app.route('/process', methods=['POST'])
def process():
    # Recebe a escolha do frontend (face ou hand)
    choice = request.json.get('choice')
    print("Escolha recebida", choice)

    if choice == 'face':
        result = detect_facial_expression()
        return jsonify({"result": result})
    elif choice == 'hand':
        result = count_fingers()
        return jsonify({"result": result})
    else:
        return jsonify({"error": "Invalid choice"}), 400

if __name__ == '__main__':
    app.run(debug=True)
