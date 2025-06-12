from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os, uuid, cv2, base64

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = YOLO('best.pt')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'ok', 'message': 'Olive Leaf Detection API is running'})

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS': return '', 200
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    min_confidence = 0.7
    results = model(filepath, conf=min_confidence)
    
    boxes = results[0].boxes
    cls_list = boxes.cls.tolist() if boxes.cls is not None else []
    conf_list = boxes.conf.tolist() if boxes.conf is not None else []
    
    leaf_details = [
        {'class_name': model.names[int(cls)], 'confidence': round(conf * 100, 2)}
        for cls, conf in zip(cls_list, conf_list)
    ]
    
    result_img = results[0].plot()
    note = 'Detections above threshold' if leaf_details else 'No detections found'
    
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    os.remove(filepath)
    return jsonify({
        'image': img_base64,
        'detection_info': {
            'leaf_count': len(leaf_details),
            'leaves': leaf_details,
            'note': note
        }
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
