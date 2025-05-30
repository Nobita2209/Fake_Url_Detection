from flask import Flask, request

app = Flask(__name__)

@app.route('/capture_url', methods=['POST'])
def capture_url():
    data = request.get_json()
    url = data.get('url')  # Extract the URL from the request data
    if url:
        print("Captured URL:", url)
        # Here you can perform any actions with the captured URL
        # For example, you can save it to a file, process it, etc.
        return 'URL captured successfully', 200
    else:
        return 'No URL provided in the request', 400

if __name__ == '__main__':
    app.run(debug=True)
