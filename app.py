"""
Flask backend for FNN-ITM web application
"""

from flask import Flask, request, jsonify, render_template
from fnn_itm import FNNITMFramework, InfluencerProfile
import os, json, traceback

app = Flask(__name__)
#@app.route('/')
#def home():
#    # Looks for 'index.html' inside a folder named 'templates'
#    return render_template('index.html') 

if __name__ == '__main__':
    # Render provides a $PORT environment variable you must bind to
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
framework = FNNITMFramework()

SAMPLE_PROFILES = {
    "high_trust": {
        "name": "Alex Rivera (High Trust Example)",
        "sentiment_polarity":    0.82,
        "topic_coherence":       0.88,
        "linguistic_quality":    0.79,
        "visual_aesthetics":     0.85,
        "ethical_tone":          0.90,
        "likes_follower_ratio":  0.75,
        "comment_depth":         0.80,
        "share_frequency":       0.72,
        "comment_authenticity":  0.88,
        "posting_frequency":     0.83,
        "follower_growth_rate":  0.70,
        "interaction_latency":   0.78,
        "engagement_volatility": 0.15,
        "audience_diversity":    0.76,
        "network_centrality":    0.71,
        "follower_reciprocity":  0.82,
    },
    "moderate_trust": {
        "name": "Jordan Blake (Moderate Trust Example)",
        "sentiment_polarity":    0.55,
        "topic_coherence":       0.50,
        "linguistic_quality":    0.48,
        "visual_aesthetics":     0.60,
        "ethical_tone":          0.52,
        "likes_follower_ratio":  0.65,
        "comment_depth":         0.42,
        "share_frequency":       0.55,
        "comment_authenticity":  0.40,
        "posting_frequency":     0.45,
        "follower_growth_rate":  0.70,
        "interaction_latency":   0.38,
        "engagement_volatility": 0.55,
        "audience_diversity":    0.50,
        "network_centrality":    0.48,
        "follower_reciprocity":  0.44,
    },
    "low_trust": {
        "name": "Sam Flux (Low Trust Example)",
        "sentiment_polarity":    0.22,
        "topic_coherence":       0.20,
        "linguistic_quality":    0.25,
        "visual_aesthetics":     0.30,
        "ethical_tone":          0.15,
        "likes_follower_ratio":  0.85,
        "comment_depth":         0.10,
        "share_frequency":       0.88,
        "comment_authenticity":  0.08,
        "posting_frequency":     0.15,
        "follower_growth_rate":  0.95,
        "interaction_latency":   0.12,
        "engagement_volatility": 0.90,
        "audience_diversity":    0.18,
        "network_centrality":    0.22,
        "follower_reciprocity":  0.10,
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        profile = InfluencerProfile(
            sentiment_polarity    = float(data.get("sentiment_polarity", 0.5)),
            topic_coherence       = float(data.get("topic_coherence", 0.5)),
            linguistic_quality    = float(data.get("linguistic_quality", 0.5)),
            visual_aesthetics     = float(data.get("visual_aesthetics", 0.5)),
            ethical_tone          = float(data.get("ethical_tone", 0.5)),
            likes_follower_ratio  = float(data.get("likes_follower_ratio", 0.5)),
            comment_depth         = float(data.get("comment_depth", 0.5)),
            share_frequency       = float(data.get("share_frequency", 0.5)),
            comment_authenticity  = float(data.get("comment_authenticity", 0.5)),
            posting_frequency     = float(data.get("posting_frequency", 0.5)),
            follower_growth_rate  = float(data.get("follower_growth_rate", 0.5)),
            interaction_latency   = float(data.get("interaction_latency", 0.5)),
            engagement_volatility = float(data.get("engagement_volatility", 0.5)),
            audience_diversity    = float(data.get("audience_diversity", 0.5)),
            network_centrality    = float(data.get("network_centrality", 0.5)),
            follower_reciprocity  = float(data.get("follower_reciprocity", 0.5)),
        )

        framework.reset_temporal_state()
        result = framework.evaluate(profile)

        return jsonify({
            "success": True,
            "final_score":             result.final_score,
            "trust_level":             result.trust_level,
            "confidence":              result.confidence,
            "content_credibility":     result.content_credibility,
            "engagement_authenticity": result.engagement_authenticity,
            "behavioral_stability":    result.behavioral_stability,
            "sentiment_community":     result.sentiment_community,
            "active_rules": [
                {
                    "rule_id":     r["rule_id"],
                    "description": r["description"],
                    "strength":    r["strength"],
                    "consequent":  r["consequent"],
                }
                for r in sorted(result.active_rules, key=lambda r: r["strength"], reverse=True)[:10]
            ],
            "fuzzy_features": [
                {
                    "name":   f.name,
                    "low":    round(f.low, 3),
                    "medium": round(f.medium, 3),
                    "high":   round(f.high, 3),
                }
                for f in result.fuzzy_features
            ],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/samples", methods=["GET"])
def samples():
    return jsonify(SAMPLE_PROFILES)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
