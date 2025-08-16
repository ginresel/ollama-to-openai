from flask import Flask, request, jsonify, Response
import requests
import os
import uuid
import time

app = Flask(__name__)

OLLAMA_API = "https://ollama.com/api/chat"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()

    # Get OpenAI-style input
    messages = data.get("messages", [])
    model = data.get("model", "gpt-oss:20b")
    stream = data.get("stream", False)

    # Forward user's own API key
    auth_header = request.headers.get("Authorization", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header,
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    resp = requests.post(OLLAMA_API, headers=headers, json=payload, stream=stream)

    if stream:
        # Stream Ollama → OpenAI-style SSE
        def generate():
            for line in resp.iter_lines():
                if line:
                    try:
                        ollama_chunk = line.decode("utf-8")
                        yield f"data: {ollama_chunk}\n\n"
                    except Exception:
                        continue
            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")

    else:
        ollama_json = resp.json()

        # Convert Ollama → OpenAI response
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": ollama_json.get("message", {}),
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
