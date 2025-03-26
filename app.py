from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from main import ResearchSystem  
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

research_system = ResearchSystem()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Research System is running!",
        "endpoints": {
            "/research": "POST with JSON {'topic': 'your_research_topic'}"
        }
    })

@app.route("/research", methods=["POST"])
async def research():
    data = request.get_json()
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        logger.info(f"Starting research on topic: {topic}")
        result = await research_system.process_query(topic)
        
        logger.info(f"Research completed for topic: {topic}")
        return jsonify({
            "status": "success",
            "topic": topic,
            "score": result["score"],
            "top_paper": result["top_paper"],
            "related_papers": result["all_papers"],
            "insights": result["insights"],
            "feedback": result["feedback"],
            "agent_status": result["agent_status"]
        })

    except Exception as e:
        logger.error(f"Research failed for topic {topic}: {str(e)}")
        return jsonify({
            "error": f"Research failed: {str(e)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)