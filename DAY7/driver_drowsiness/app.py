from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

# Open webcam
camera = cv2.VideoCapture(0)

# Class names (IMPORTANT for filtering)
classNames = [
    "closed_eye",
    "closed_eye_yawning_FACE",
    "distracted",
    "mobile",
    "mobile_usage",
    "no_distraction",
    "no_yawning_face",
    "normal_face",
    "open_eye",
    "slightly_closed_eye",
    "slightly_closed_eye_yawning_face",
    "tried_face",
    "yawning_face"
]


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLO prediction
        results = model(frame)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # confidence
                conf = float(box.conf[0])

                # class
                cls = int(box.cls[0])
                label = classNames[cls]

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/monitor")
def monitor():
    return render_template("monitor.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)