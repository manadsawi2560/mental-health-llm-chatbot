import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag_chain import answer

load_dotenv()
app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/chat")
def chat():
    try:
        data = request.get_json(force=True)
        msg = (data.get("message") or "").strip()
        if not msg:
            return jsonify({"ok": False, "error": "Empty message"}), 400
        resp = answer(msg)
        footer = "\n\n— ข้อจำกัด: บอทนี้ให้ข้อมูลทั่วไป ไม่ใช่การวินิจฉัยโรค หากมีอาการน่ากังวลควรพบแพทย์หรือโทรฉุกเฉิน"
        return jsonify({"ok": True, "reply": resp + footer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
