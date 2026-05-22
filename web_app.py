import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from traffic_detection import (
    analyze_dataset_image,
    analyze_dataset_image_retry,
    analyze_simple_image,
    annotate_frame,
)

HOST = "127.0.0.1"
PORT = 8000

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Traffic Sign Detector</title>
  <style>
    :root {
      --bg: #f4efe6;
      --card: #fffaf2;
      --ink: #1f1f1f;
      --accent: #0d6b5f;
      --accent-2: #cc4b1f;
      --line: #d9cfbf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(204,75,31,0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(13,107,95,0.18), transparent 24%),
        var(--bg);
      color: var(--ink);
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3rem);
      letter-spacing: 0.02em;
    }
    p.lead {
      margin: 0 0 22px;
      max-width: 760px;
      line-height: 1.5;
      color: #4b463f;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(68, 56, 38, 0.08);
    }
    h2 {
      margin: 0 0 12px;
      font-size: 1.25rem;
    }
    .controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    button, input[type=file]::file-selector-button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button.alt {
      background: var(--accent-2);
    }
    input[type=file] {
      width: 100%;
      font: inherit;
    }
    .stage {
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 14px;
      border: 1px dashed var(--line);
      overflow: hidden;
      background: #ece6da;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .stage img, .stage video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: black;
    }
    .result {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(13,107,95,0.08);
      min-height: 58px;
      line-height: 1.4;
    }
    .muted { color: #6c655b; }
    .badge {
      display: inline-block;
      margin-top: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(204,75,31,0.12);
      color: #7e2d10;
      font-size: 0.95rem;
    }
    canvas { display: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Traffic Sign Detector</h1>
    <p class="lead">Upload one sign image or use your camera. The detector analyzes the original color image, crops the most sign-like region, and returns the predicted result with an annotated preview.</p>
    <div class="grid">
      <section class="card">
        <h2>Upload Photo</h2>
        <div class="controls">
          <input id="fileInput" type="file" accept="image/*">
          <button id="analyzeUpload">Analyze Sign</button>
          <button id="retryUpload" class="alt">Analyze Again</button>
        </div>
        <div class="stage"><img id="uploadPreview" alt="Upload preview"></div>
        <div id="uploadResult" class="result muted">Choose a photo to analyze.</div>
      </section>
      <section class="card">
        <h2>Camera Mode</h2>
        <div class="controls">
          <button id="startCamera">Open Camera</button>
          <button id="analyzeCamera">Analyze Current Frame</button>
          <button id="stopCamera" class="alt">Stop Camera</button>
        </div>
        <div class="stage"><video id="camera" autoplay playsinline muted></video></div>
        <div id="cameraResult" class="result muted">Open the camera and show one sign clearly on paper or on your phone screen.</div>
        <span class="badge">Best with one large centered sign</span>
      </section>
    </div>
    <canvas id="canvas"></canvas>
  </div>
  <script>
    const fileInput = document.getElementById("fileInput");
    const analyzeUploadButton = document.getElementById("analyzeUpload");
    const retryUploadButton = document.getElementById("retryUpload");
    const uploadPreview = document.getElementById("uploadPreview");
    const uploadResult = document.getElementById("uploadResult");
    const video = document.getElementById("camera");
    const cameraResult = document.getElementById("cameraResult");
    const startButton = document.getElementById("startCamera");
    const analyzeCameraButton = document.getElementById("analyzeCamera");
    const stopButton = document.getElementById("stopCamera");
    const canvas = document.getElementById("canvas");
    let stream = null;
    let uploadDataUrl = null;
    let cameraBusy = false;

    async function analyzeDataUrl(dataUrl, mode) {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl, mode })
      });
      if (!response.ok) {
        throw new Error("Analysis failed");
      }
      return response.json();
    }

    function renderResult(target, data) {
      if (!data.detected) {
        target.innerHTML = "No sign detected.";
        return;
      }
      target.innerHTML = `<strong>${data.name}</strong><br>Confidence: ${data.confidence.toFixed(1)}%`;
    }

    fileInput.addEventListener("change", async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        uploadDataUrl = reader.result;
        uploadPreview.src = uploadDataUrl;
        uploadResult.textContent = "Image ready. Press Analyze Sign.";
      };
      reader.readAsDataURL(file);
    });

    analyzeUploadButton.addEventListener("click", async () => {
      if (!uploadDataUrl) {
        uploadResult.textContent = "Choose an image first.";
        return;
      }
      uploadResult.textContent = "Analyzing...";
      try {
        const data = await analyzeDataUrl(uploadDataUrl, "upload");
        uploadPreview.src = data.annotated_image;
        renderResult(uploadResult, data);
      } catch (error) {
        uploadResult.textContent = "Could not analyze that image.";
      }
    });

    retryUploadButton.addEventListener("click", async () => {
      if (!uploadDataUrl) {
        uploadResult.textContent = "Choose an image first.";
        return;
      }
      uploadResult.textContent = "Re-analyzing...";
      try {
        const data = await analyzeDataUrl(uploadDataUrl, "upload_retry");
        uploadPreview.src = data.annotated_image;
        renderResult(uploadResult, data);
      } catch (error) {
        uploadResult.textContent = "Could not analyze that image.";
      }
    });

    startButton.addEventListener("click", async () => {
      if (stream) return;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        video.srcObject = stream;
        cameraResult.textContent = "Camera running. Hold one sign clearly, then press Analyze Current Frame.";
      } catch (error) {
        cameraResult.textContent = "Could not access the camera.";
      }
    });

    analyzeCameraButton.addEventListener("click", async () => {
      if (!stream || video.readyState < 2 || cameraBusy) {
        cameraResult.textContent = "Open the camera first.";
        return;
      }
      cameraBusy = true;
      cameraResult.textContent = "Analyzing camera frame...";
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      try {
        const data = await analyzeDataUrl(dataUrl, "camera");
        renderResult(cameraResult, data);
      } catch (error) {
        cameraResult.textContent = "Camera analysis failed.";
      } finally {
        cameraBusy = false;
      }
    });

    stopButton.addEventListener("click", () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
      cameraBusy = false;
      cameraResult.textContent = "Camera stopped.";
    });
  </script>
</body>
</html>
"""


def decode_data_url(data_url):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def encode_image(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def analyze_image_frame(frame, mode="upload"):
    if mode == "upload_retry":
        detection = analyze_dataset_image_retry(frame)
    elif mode == "upload":
        detection = analyze_dataset_image(frame)
    else:
        detection = analyze_simple_image(frame)
    annotated = frame.copy()

    if detection is not None:
        annotate_frame(
            annotated,
            detection["box"],
            detection["name"],
            detection["confidence"],
            detection["secondary_name"],
            detection["secondary_confidence"],
        )

    return {
        "detected": detection is not None,
        "name": detection["name"] if detection is not None else None,
        "confidence": detection["confidence"] if detection is not None else 0.0,
        "ocr_speed": detection.get("ocr_speed") if detection is not None else None,
        "ocr_confidence": detection.get("ocr_confidence") if detection is not None else None,
        "candidate_count": 1 if detection is not None else 0,
        "annotated_image": encode_image(annotated),
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body.decode("utf-8"))
            frame = decode_data_url(payload["image"])
            result = analyze_image_frame(frame, payload.get("mode", "upload"))
            response = json.dumps(result).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except Exception as exc:
            message = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)


def main():
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Open http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
