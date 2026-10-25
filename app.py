import base64
import json
import os
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS

from largest_rect import process

app = Flask(__name__)
CORS(app)  # lets the Flutter web/debug build call this from a different origin


@app.get("/")
def health():
    return "ok"


@app.post("/process")
def process_endpoint():
    if "image" not in request.files:
        return jsonify(error="send the image as multipart form field 'image'"), 400

    upload = request.files["image"]

    # process entirely via scratch files that get removed once the
    # response is built -- nothing from the request is kept on disk
    in_fd, in_path = tempfile.mkstemp(suffix=os.path.splitext(upload.filename or "")[1] or ".jpg")
    out_fd, out_path = tempfile.mkstemp(suffix=".png")
    os.close(in_fd)
    os.close(out_fd)
    json_path = f"{os.path.splitext(out_path)[0]}.json"
    try:
        upload.save(in_path)
        try:
            box, circle_count, column_count, filled_count = process(in_path, out_path, json_path)
        except (FileNotFoundError, ValueError) as e:
            return jsonify(error=str(e)), 400

        with open(out_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        with open(json_path) as f:
            results = json.load(f)
    finally:
        for path in (in_path, out_path, json_path):
            if os.path.exists(path):
                os.remove(path)

    return jsonify(
        box=box,
        circle_count=circle_count,
        column_count=column_count,
        filled_count=int(filled_count),
        results=results,
        image_base64=image_b64,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
