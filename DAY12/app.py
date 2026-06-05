from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
closed_eye_count = 0
yawning_count = 0
mobile_count = 0
distracted_count = 0
app = Flask(__name__)

# Load model
model = YOLO("best.pt")

# Open webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
    global closed_eye_count
    global yawning_count
    global mobile_count
    global distracted_count
    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLO prediction
        results = model(frame, conf=0.20)

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
                if label == "closed_eye":
                    closed_eye_count += 1

                elif label == "yawning_face":
                    yawning_count += 1

                elif label == "mobile_usage":
                    mobile_count += 1

                elif label == "distracted":
                    distracted_count += 1

                # print prediction in terminal
                print(label, conf)

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
        # Display counters
        cv2.putText(frame, f"Closed Eye: {closed_eye_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Yawning: {yawning_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Mobile: {mobile_count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Distracted: {distracted_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
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
@app.route("/report")
def report():
    return render_template(
        "report.html",
        closed_eye=closed_eye_count,
        yawning=yawning_count,
        mobile=mobile_count,
        distracted=distracted_count
    )


if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5000, debug=False)