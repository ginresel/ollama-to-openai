from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OLLAMA_API = "https://ollama.com/api/chat"  # Turbo endpoint
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()

    # Convert OpenAI-style -> Ollama-style
    messages = data.get("messages", [])
    model = data.get("model", "gpt-oss:20b")
    stream = data.get("stream", False)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_KEY}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    resp = requests.post(OLLAMA_API, headers=headers, json=payload, stream=stream)

    if stream:
        # Stream Ollama chunks back as OpenAI-style chunks
        def generate():
            for line in resp.iter_lines():
                if line:
                    yield f"data: {line.decode('utf-8')}\n\n"
            yield "data: [DONE]\n\n"

        return app.response_class(generate(), mimetype="text/event-stream")

    else:
        ollama_json = resp.json()

        # Convert Ollama response -> OpenAI response shape
        return jsonify({
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": ollama_json.get("message", {}),
                "finish_reason": "stop"
            }],
            "model": model,
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
