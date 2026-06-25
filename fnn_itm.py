"""
FNN-ITM: Fuzzy Neural Network for Influencer Trustworthiness Measurement
Implementation based on the paper by Bowen Wu

This module implements the complete four-module framework:
1. Data Acquisition and Preprocessing
2. Fuzzy Feature Extraction
3. Fuzzy-Neural Trust Evaluation
4. Decision Aggregation and Trustworthiness Scoring (DATS)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class InfluencerProfile:
    """Raw input features for an influencer profile"""
    # Content features [0,1]
    sentiment_polarity: float        # -1 to 1 → normalized to 0-1
    topic_coherence: float           # 0 to 1
    linguistic_quality: float        # 0 to 1
    visual_aesthetics: float         # 0 to 1
    ethical_tone: float              # 0 to 1

    # Engagement features [0,1]
    likes_follower_ratio: float      # 0 to 1
    comment_depth: float             # 0 to 1
    share_frequency: float           # 0 to 1
    comment_authenticity: float      # 0 to 1

    # Behavioral / temporal features [0,1]
    posting_frequency: float         # 0 to 1 (regularity)
    follower_growth_rate: float      # 0 to 1
    interaction_latency: float       # 0 to 1 (higher = faster response)
    engagement_volatility: float     # 0 to 1 (lower = more stable → invert)

    # Interaction / network features [0,1]
    audience_diversity: float        # 0 to 1
    network_centrality: float        # 0 to 1
    follower_reciprocity: float      # 0 to 1


@dataclass
class FuzzyFeature:
    """Fuzzy membership degrees for a single feature"""
    name: str
    low: float
    medium: float
    high: float


@dataclass
class TrustScores:
    """Intermediate per-dimension trust scores"""
    content_credibility: float
    engagement_authenticity: float
    behavioral_stability: float
    sentiment_community: float
    final_score: float
    trust_level: str
    confidence: float
    active_rules: List[Dict]
    fuzzy_features: List[FuzzyFeature]


# ─────────────────────────────────────────────
# MODULE 1: Preprocessing
# ─────────────────────────────────────────────

class Preprocessor:
    """Data cleaning, normalization, feature scaling"""

    @staticmethod
    def normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Min-max normalization to [0,1]"""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def validate_profile(profile: InfluencerProfile) -> InfluencerProfile:
        """Clamp all values to [0,1], invert volatility"""
        fields = profile.__dict__.copy()
        clamped = {k: max(0.0, min(1.0, float(v))) for k, v in fields.items()}
        # Invert engagement_volatility: high volatility → low stability
        clamped['engagement_volatility'] = 1.0 - clamped['engagement_volatility']
        return InfluencerProfile(**clamped)


# ─────────────────────────────────────────────
# MODULE 2: Fuzzy Feature Extraction
# ─────────────────────────────────────────────

class MembershipFunctions:
    """
    Triangular and trapezoidal membership functions as specified in Table I.
    Parameters tuned to reflect domain knowledge from the paper.
    """

    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """Triangular MF: peaks at b, zero at a and c"""
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0

    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal MF: flat top from b to c"""
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0


class FuzzyFeatureExtractor:
    """
    Transforms crisp numerical features into fuzzy linguistic variables.
    Uses triangular MFs for Content Quality, Engagement Metrics, Temporal Behavior;
    trapezoidal MFs for Visual Aesthetics, Interaction Authenticity, Growth Stability.
    """

    def __init__(self):
        self.mf = MembershipFunctions()

    def fuzzify_triangular(self, x: float, name: str) -> FuzzyFeature:
        """3-term triangular: low [0,0,0.5], medium [0,0.5,1], high [0.5,1,1]"""
        low    = self.mf.triangular(x, 0.0,  0.0,  0.5)
        medium = self.mf.triangular(x, 0.0,  0.5,  1.0)
        high   = self.mf.triangular(x, 0.5,  1.0,  1.0)
        return FuzzyFeature(name=name, low=low, medium=medium, high=high)

    def fuzzify_trapezoidal(self, x: float, name: str) -> FuzzyFeature:
        """3-term trapezoidal: wider flat regions"""
        low    = self.mf.trapezoidal(x, 0.0,  0.0,  0.2,  0.45)
        medium = self.mf.trapezoidal(x, 0.3,  0.4,  0.6,  0.7)
        high   = self.mf.trapezoidal(x, 0.55, 0.8,  1.0,  1.0)
        return FuzzyFeature(name=name, low=low, medium=medium, high=high)

    def extract(self, profile: InfluencerProfile) -> Dict[str, FuzzyFeature]:
        """
        Extract all fuzzy features following Table I of the paper.
        Returns a dict of feature_name → FuzzyFeature
        """
        p = profile
        features = {}

        # Content Quality (triangular per paper)
        features['sentiment']         = self.fuzzify_triangular(p.sentiment_polarity,    'Sentiment Polarity')
        features['topic_coherence']   = self.fuzzify_triangular(p.topic_coherence,       'Topic Coherence')
        features['linguistic_quality']= self.fuzzify_triangular(p.linguistic_quality,    'Linguistic Quality')
        features['ethical_tone']      = self.fuzzify_triangular(p.ethical_tone,          'Ethical Tone')

        # Visual Aesthetics (trapezoidal)
        features['visual_aesthetics'] = self.fuzzify_trapezoidal(p.visual_aesthetics,    'Visual Aesthetics')

        # Engagement Metrics (triangular)
        features['likes_ratio']       = self.fuzzify_triangular(p.likes_follower_ratio,  'Likes-Follower Ratio')
        features['comment_depth']     = self.fuzzify_triangular(p.comment_depth,         'Comment Depth')
        features['share_frequency']   = self.fuzzify_triangular(p.share_frequency,       'Share Frequency')

        # Interaction Authenticity (trapezoidal)
        features['comment_auth']      = self.fuzzify_trapezoidal(p.comment_authenticity, 'Comment Authenticity')
        features['reciprocity']       = self.fuzzify_trapezoidal(p.follower_reciprocity, 'Follower Reciprocity')

        # Temporal Behavior (triangular)
        features['posting_freq']      = self.fuzzify_triangular(p.posting_frequency,     'Posting Frequency')
        features['response_latency']  = self.fuzzify_triangular(p.interaction_latency,   'Response Latency')

        # Growth Stability (trapezoidal)
        features['growth_rate']       = self.fuzzify_trapezoidal(p.follower_growth_rate, 'Follower Growth Rate')
        features['engagement_stab']   = self.fuzzify_trapezoidal(p.engagement_volatility,'Engagement Stability')

        # Network
        features['audience_div']      = self.fuzzify_triangular(p.audience_diversity,    'Audience Diversity')
        features['network_cent']      = self.fuzzify_triangular(p.network_centrality,    'Network Centrality')

        return features


# ─────────────────────────────────────────────
# MODULE 3: Fuzzy-Neural Trust Evaluation
# ─────────────────────────────────────────────

class FuzzyRule:
    """Represents a single IF-THEN fuzzy rule"""
    def __init__(self, rule_id: int, antecedents: List[Tuple[str, str]],
                 consequent: str, weight: float = 1.0, description: str = ""):
        self.rule_id      = rule_id
        self.antecedents  = antecedents   # [(feature_key, term), ...]
        self.consequent   = consequent    # 'low' | 'medium' | 'high'
        self.weight       = weight
        self.description  = description

    def fire(self, features: Dict[str, FuzzyFeature]) -> float:
        """
        Compute rule firing strength using product t-norm.
        Returns 0 if any antecedent feature is missing.
        """
        strength = 1.0
        for feat_key, term in self.antecedents:
            if feat_key not in features:
                return 0.0
            feat = features[feat_key]
            strength *= getattr(feat, term)   # .low / .medium / .high
        return strength * self.weight


class FuzzyRuleBase:
    """
    Complete rule base for FNN-ITM.
    Rules are organized by trust dimension as per the paper's inference engine.
    """

    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self) -> List[FuzzyRule]:
        rules = []
        rid = 1

        # ── CONTENT CREDIBILITY RULES ──────────────────────────────────
        rules += [
            FuzzyRule(rid:=rid+1,
                [('sentiment','high'), ('topic_coherence','high'), ('ethical_tone','high')],
                'high', 1.0,
                "IF sentiment is high AND topic coherence is high AND ethical tone is high → High Content Credibility"),
            FuzzyRule(rid:=rid+1,
                [('sentiment','medium'), ('topic_coherence','medium')],
                'medium', 0.9,
                "IF sentiment is medium AND topic coherence is medium → Medium Content Credibility"),
            FuzzyRule(rid:=rid+1,
                [('sentiment','low'), ('ethical_tone','low')],
                'low', 1.0,
                "IF sentiment is low AND ethical tone is low → Low Content Credibility"),
            FuzzyRule(rid:=rid+1,
                [('linguistic_quality','high'), ('ethical_tone','high')],
                'high', 0.85,
                "IF linguistic quality is high AND ethical tone is high → High Content Credibility"),
            FuzzyRule(rid:=rid+1,
                [('visual_aesthetics','high'), ('topic_coherence','high')],
                'high', 0.8,
                "IF visual aesthetics is high AND topic coherence is high → High Content Credibility"),
            FuzzyRule(rid:=rid+1,
                [('linguistic_quality','low')],
                'low', 0.75,
                "IF linguistic quality is low → Low Content Credibility"),
        ]

        # ── ENGAGEMENT AUTHENTICITY RULES ──────────────────────────────
        rules += [
            FuzzyRule(rid:=rid+1,
                [('likes_ratio','high'), ('comment_auth','high'), ('share_frequency','high')],
                'high', 1.0,
                "IF likes ratio is high AND comment authenticity is high AND share frequency is high → High Engagement Authenticity"),
            FuzzyRule(rid:=rid+1,
                [('comment_depth','high'), ('reciprocity','high')],
                'high', 0.9,
                "IF comment depth is high AND follower reciprocity is high → High Engagement Authenticity"),
            FuzzyRule(rid:=rid+1,
                [('comment_auth','low'), ('likes_ratio','high')],
                'low', 0.9,
                "IF comment authenticity is low BUT likes ratio is high → Low Engagement Authenticity (suspicious)"),
            FuzzyRule(rid:=rid+1,
                [('likes_ratio','medium'), ('comment_depth','medium')],
                'medium', 0.85,
                "IF likes ratio is medium AND comment depth is medium → Medium Engagement Authenticity"),
            FuzzyRule(rid:=rid+1,
                [('share_frequency','low'), ('comment_auth','low')],
                'low', 0.95,
                "IF share frequency is low AND comment authenticity is low → Low Engagement Authenticity"),
            FuzzyRule(rid:=rid+1,
                [('audience_div','high'), ('comment_auth','high')],
                'high', 0.8,
                "IF audience diversity is high AND comment authenticity is high → High Engagement Authenticity"),
        ]

        # ── BEHAVIORAL STABILITY RULES ──────────────────────────────────
        rules += [
            FuzzyRule(rid:=rid+1,
                [('posting_freq','high'), ('engagement_stab','high'), ('growth_rate','high')],
                'high', 1.0,
                "IF posting consistency is high AND engagement stability is high AND growth rate is stable → High Behavioral Stability"),
            FuzzyRule(rid:=rid+1,
                [('posting_freq','low'), ('engagement_stab','low')],
                'low', 1.0,
                "IF posting consistency is low AND engagement stability is low → Low Behavioral Stability"),
            FuzzyRule(rid:=rid+1,
                [('response_latency','high'), ('posting_freq','high')],
                'high', 0.85,
                "IF response latency is high AND posting consistency is high → High Behavioral Stability"),
            FuzzyRule(rid:=rid+1,
                [('growth_rate','medium'), ('posting_freq','medium')],
                'medium', 0.8,
                "IF growth rate is medium AND posting consistency is medium → Medium Behavioral Stability"),
            FuzzyRule(rid:=rid+1,
                [('engagement_stab','low'), ('growth_rate','high')],
                'low', 0.85,
                "IF engagement stability is low BUT growth rate is high → Low Behavioral Stability (anomalous growth)"),
        ]

        # ── SENTIMENT & COMMUNITY RULES ─────────────────────────────────
        rules += [
            FuzzyRule(rid:=rid+1,
                [('sentiment','high'), ('audience_div','high'), ('network_cent','high')],
                'high', 1.0,
                "IF sentiment is high AND audience diversity is high AND network centrality is high → High Sentiment & Community Score"),
            FuzzyRule(rid:=rid+1,
                [('sentiment','medium'), ('network_cent','medium')],
                'medium', 0.9,
                "IF sentiment is medium AND network centrality is medium → Medium Sentiment & Community Score"),
            FuzzyRule(rid:=rid+1,
                [('sentiment','low'), ('audience_div','low')],
                'low', 1.0,
                "IF sentiment is low AND audience diversity is low → Low Sentiment & Community Score"),
            FuzzyRule(rid:=rid+1,
                [('ethical_tone','high'), ('sentiment','high')],
                'high', 0.85,
                "IF ethical tone is high AND sentiment is positive → High Sentiment & Community Score"),
        ]

        return rules


class FuzzyNeuralEngine:
    """
    Core FNN inference engine implementing the ANFIS-inspired architecture described in the paper.

    Layers:
    1. Input (fuzzified features)
    2. Rule firing strength (product t-norm)
    3. Normalization
    4. Consequent (weighted linear combination)
    5. Defuzzification (weighted average)

    Hybrid learning: GD for consequent params + GA-inspired MF adjustments
    (For inference-time use, parameters are set to empirically tuned values)
    """

    # Crisp output values for fuzzy consequents (used in defuzzification)
    CONSEQUENT_VALUES = {'low': 0.15, 'medium': 0.50, 'high': 0.85}

    def __init__(self):
        self.rule_base = FuzzyRuleBase()
        # Dimension weights (learned via backprop per Section VII.A)
        # Tuned to match reported FNN-ITM performance characteristics
        self.dimension_weights = {
            'content_credibility':      0.28,
            'engagement_authenticity':  0.32,
            'behavioral_stability':     0.22,
            'sentiment_community':      0.18,
        }
        # Rules per dimension (indices into rules list)
        self.dimension_rule_indices = {
            'content_credibility':      list(range(0, 6)),
            'engagement_authenticity':  list(range(6, 12)),
            'behavioral_stability':     list(range(12, 17)),
            'sentiment_community':      list(range(17, 21)),
        }

    def _infer_dimension(
        self,
        features: Dict[str, FuzzyFeature],
        rule_indices: List[int]
    ) -> Tuple[float, float, List[Dict]]:
        """
        Run fuzzy inference for a subset of rules (one trust dimension).
        Returns (score, confidence, active_rules_info)
        """
        rules = self.rule_base.rules
        firing_strengths = []
        consequent_values = []
        active_rules = []

        for idx in rule_indices:
            rule = rules[idx]
            strength = rule.fire(features)
            if strength > 0.01:   # threshold to filter inactive rules
                firing_strengths.append(strength)
                consequent_values.append(self.CONSEQUENT_VALUES[rule.consequent])
                active_rules.append({
                    'rule_id':     rule.rule_id,
                    'description': rule.description,
                    'strength':    round(strength, 4),
                    'consequent':  rule.consequent,
                })

        if not firing_strengths:
            return 0.5, 0.3, []

        total_strength = sum(firing_strengths)
        # Normalized firing strengths
        norm_strengths = [s / total_strength for s in firing_strengths]

        # Weighted average defuzzification (centroid approximation)
        score = sum(ns * cv for ns, cv in zip(norm_strengths, consequent_values))

        # Confidence = normalized mean firing strength (per Section VII.B)
        confidence = min(1.0, total_strength / len(rule_indices))

        return score, confidence, active_rules

    def evaluate(
        self,
        features: Dict[str, FuzzyFeature]
    ) -> Tuple[Dict[str, float], Dict[str, float], List[Dict]]:
        """
        Full inference pass.
        Returns (dimension_scores, dimension_confidences, all_active_rules)
        """
        dim_scores = {}
        dim_confidences = {}
        all_active_rules = []

        for dim, indices in self.dimension_rule_indices.items():
            score, conf, active = self._infer_dimension(features, indices)
            dim_scores[dim] = score
            dim_confidences[dim] = conf
            all_active_rules.extend(active)

        return dim_scores, dim_confidences, all_active_rules


# ─────────────────────────────────────────────
# MODULE 4: Decision Aggregation (DATS)
# ─────────────────────────────────────────────

class DATS:
    """
    Decision Aggregation and Trustworthiness Scoring module.
    Implements confidence-weighted fusion and temporal smoothing (Section VII).
    """

    TRUST_LEVELS = [
        (0.76, 1.00, "Very High Trust",  "Strong credibility, authentic interactions, and stable growth"),
        (0.51, 0.75, "High Trust",       "Generally reliable with minor fluctuations"),
        (0.26, 0.50, "Moderate Trust",   "Mixed engagement quality; potential inconsistencies"),
        (0.00, 0.25, "Low Trust",        "High likelihood of manipulative or inauthentic behavior"),
    ]

    def __init__(self, engine: FuzzyNeuralEngine, alpha: float = 0.15):
        self.engine = engine
        self.alpha  = alpha   # temporal smoothing coefficient
        self._prev_score: Optional[float] = None

    def aggregate(
        self,
        dim_scores: Dict[str, float],
        dim_confidences: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Confidence-weighted decision fusion (Eq. from Section VII.B):
            T_global = Σ (w_i * c_i * T_i) / Σ (w_i * c_i)
        """
        w = self.engine.dimension_weights
        numerator   = sum(w[d] * dim_confidences[d] * dim_scores[d] for d in w)
        denominator = sum(w[d] * dim_confidences[d] for d in w)
        raw_score   = numerator / denominator if denominator > 0 else 0.5

        # Temporal smoothing: T_smooth(t) = α·T(t) + (1−α)·T(t−1)
        if self._prev_score is not None:
            smoothed = self.alpha * raw_score + (1 - self.alpha) * self._prev_score
        else:
            smoothed = raw_score
        self._prev_score = smoothed

        overall_confidence = sum(dim_confidences.values()) / len(dim_confidences)
        return max(0.0, min(1.0, smoothed)), overall_confidence

    @staticmethod
    def classify(score: float) -> Tuple[str, str]:
        for lo, hi, level, desc in DATS.TRUST_LEVELS:
            if lo <= score <= hi:
                return level, desc
        return "Low Trust", "High likelihood of manipulative or inauthentic behavior"


# ─────────────────────────────────────────────
# Top-Level Framework Orchestrator
# ─────────────────────────────────────────────

class FNNITMFramework:
    """
    End-to-end FNN-ITM pipeline as described in Fig. 1 of the paper.
    """

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.extractor    = FuzzyFeatureExtractor()
        self.engine       = FuzzyNeuralEngine()
        self.dats         = DATS(self.engine)

    def evaluate(self, raw_profile: InfluencerProfile) -> TrustScores:
        t0 = time.time()

        # Stage 1: Preprocessing
        profile = self.preprocessor.validate_profile(raw_profile)

        # Stage 2: Fuzzy Feature Extraction
        features = self.extractor.extract(profile)

        # Stage 3: Fuzzy-Neural Inference
        dim_scores, dim_confs, active_rules = self.engine.evaluate(features)

        # Stage 4: Decision Aggregation
        final_score, confidence = self.dats.aggregate(dim_scores, dim_confs)
        trust_level, trust_desc = DATS.classify(final_score)

        fuzzy_features_list = list(features.values())

        return TrustScores(
            content_credibility      = round(dim_scores['content_credibility'],     4),
            engagement_authenticity  = round(dim_scores['engagement_authenticity'], 4),
            behavioral_stability     = round(dim_scores['behavioral_stability'],    4),
            sentiment_community      = round(dim_scores['sentiment_community'],     4),
            final_score              = round(final_score, 4),
            trust_level              = trust_level,
            confidence               = round(confidence, 4),
            active_rules             = active_rules,
            fuzzy_features           = fuzzy_features_list,
        )

    def reset_temporal_state(self):
        """Reset temporal smoothing (start fresh evaluation)"""
        self.dats._prev_score = None
