"""
Alignment Engine Module
Analyzes resume bullets against job description and embedding store
"""

from typing import List, Dict, Tuple
import numpy as np
from faiss import normalize_L2


class AlignmentEngine:
    """Analyzes and aligns resume bullets with job description"""
    
    def __init__(self, embedding_store, similarity_threshold: float = 0.6,
                 rewrite_threshold: float = 0.4, keep_threshold: float = 0.75):
        self.embedding_store = embedding_store
        self.similarity_threshold = similarity_threshold
        self.rewrite_threshold = rewrite_threshold
        self.keep_threshold = keep_threshold
        
        # Embed job description for comparison
        self.jd_embedding = None
        self.jd_keywords = ""
    
    def set_job_description(self, jd_keywords: str):
        """Set job description keywords for comparison"""
        self.jd_keywords = jd_keywords
        # Embed JD for similarity search
        self.jd_embedding = self.embedding_store.model.encode(
            [jd_keywords], 
            convert_to_numpy=True
        )
        # Normalize for cosine similarity
        from faiss import normalize_L2
        normalize_L2(self.jd_embedding)
    
    def analyze_bullet(self, bullet: Dict, user_profile_data: Dict = None, 
                      match_analysis: Dict = None) -> Dict:
        """
        Analyze a single bullet point
        Returns decision: KEEP, REWRITE, ADD, or DE_EMPHASIZE
        """
        bullet_text = bullet["text"]
        
        # 1. Check similarity to job description
        jd_similarity = self._calculate_jd_similarity(bullet_text)
        
        # 2. Find relevant user profile entries
        relevant_entries = self.embedding_store.get_relevant_entries(
            bullet_text, 
            threshold=self.similarity_threshold,
            top_k=5
        )
        
        # 3. Check if bullet has supporting evidence in profile
        has_evidence = len(relevant_entries) > 0
        
        # 4. Check keyword overlap
        keyword_score = self._calculate_keyword_overlap(bullet_text)
        
        # 5. Use match analysis to understand if this aligns with profile strengths
        profile_alignment = 0.0
        should_emphasize = False
        should_add = False
        
        if match_analysis:
            # Check if bullet mentions a matched skill
            for skill, evidence in match_analysis.get("skill_matches", {}).items():
                if skill.lower() in bullet_text.lower():
                    profile_alignment = 0.8
                    should_emphasize = True
                    break
            
            # Check if bullet could be enhanced based on recommendations
            for rec in match_analysis.get("recommendations", []):
                skill_or_topic = rec.get("skill_or_topic", "").lower()
                action = rec.get("action", "")
                if skill_or_topic and skill_or_topic in bullet_text.lower():
                    if action == "EMPHASIZE":
                        should_emphasize = True
                        profile_alignment = max(profile_alignment, 0.7)
                    elif action == "ADD":
                        # This might be a new bullet to add
                        should_add = True
        
        # 6. Make decision (enhanced with profile analysis)
        decision = self._make_decision(jd_similarity, has_evidence, keyword_score, 
                                      profile_alignment, should_emphasize, should_add)
        
        analysis = {
            "bullet": bullet,
            "jd_similarity": float(jd_similarity),
            "has_evidence": has_evidence,
            "relevant_entries": relevant_entries[:3],  # Top 3
            "keyword_score": float(keyword_score),
            "profile_alignment": float(profile_alignment),
            "should_emphasize": should_emphasize,
            "decision": decision,
            "reasoning": self._generate_reasoning(jd_similarity, has_evidence, keyword_score, 
                                                 decision, profile_alignment)
        }
        
        return analysis
    
    def _calculate_jd_similarity(self, bullet_text: str) -> float:
        """Calculate similarity between bullet and job description"""
        if self.jd_embedding is None:
            return 0.0
        
        # Encode bullet
        bullet_embedding = self.embedding_store.model.encode(
            [bullet_text],
            convert_to_numpy=True
        )
        normalize_L2(bullet_embedding)
        
        # Cosine similarity (dot product after normalization)
        similarity = np.dot(self.jd_embedding[0], bullet_embedding[0])
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def _calculate_keyword_overlap(self, bullet_text: str) -> float:
        """Calculate keyword overlap score (simple word matching)"""
        if not self.jd_keywords:
            return 0.0
        
        bullet_words = set(bullet_text.lower().split())
        jd_words = set(self.jd_keywords.lower().split())
        
        if not jd_words:
            return 0.0
        
        overlap = len(bullet_words.intersection(jd_words))
        return min(1.0, overlap / len(jd_words))  # Normalize
    
    def _make_decision(self, jd_similarity: float, has_evidence: bool, keyword_score: float,
                      profile_alignment: float = 0.0, should_emphasize: bool = False,
                      should_add: bool = False) -> str:
        """
        Make decision: KEEP, REWRITE, ADD, or DE_EMPHASIZE
        Enhanced with profile analysis
        """
        # Combined score with profile alignment
        combined_score = (
            jd_similarity * 0.4 + 
            keyword_score * 0.25 + 
            (0.8 if has_evidence else 0.2) * 0.2 +
            profile_alignment * 0.15
        )
        
        # If should emphasize based on profile strengths, prioritize keeping/rewriting
        if should_emphasize and has_evidence:
            if jd_similarity >= self.keep_threshold * 0.9:
                return "KEEP"
            else:
                return "REWRITE"  # Rewrite to emphasize profile strengths
        
        if should_add and not has_evidence:
            # This might be a new bullet to add, but we'll handle this separately
            pass
        
        if combined_score >= self.keep_threshold and jd_similarity >= self.keep_threshold:
            return "KEEP"
        elif combined_score >= self.rewrite_threshold:
            return "REWRITE"
        else:
            return "DE_EMPHASIZE"
    
    def _generate_reasoning(self, jd_similarity: float, has_evidence: bool, 
                           keyword_score: float, decision: str,
                           profile_alignment: float = 0.0) -> str:
        """Generate human-readable reasoning for decision"""
        reasons = []
        
        if jd_similarity >= 0.7:
            reasons.append(f"High JD similarity ({jd_similarity:.2f})")
        elif jd_similarity >= 0.5:
            reasons.append(f"Moderate JD similarity ({jd_similarity:.2f})")
        else:
            reasons.append(f"Low JD similarity ({jd_similarity:.2f})")
        
        if has_evidence:
            reasons.append("Has supporting evidence in profile")
        else:
            reasons.append("Limited evidence in profile")
        
        if keyword_score >= 0.3:
            reasons.append(f"Good keyword overlap ({keyword_score:.2f})")
        
        if profile_alignment >= 0.7:
            reasons.append(f"Strong profile alignment ({profile_alignment:.2f})")
        
        decision_reason = f"Decision: {decision} because " + "; ".join(reasons)
        
        return decision_reason
    
    def analyze_all_bullets(self, bullets: List[Dict], user_profile_data: Dict = None,
                           match_analysis: Dict = None) -> List[Dict]:
        """Analyze all bullets and return decisions, enhanced with profile match analysis"""
        analyses = []
        
        for bullet in bullets:
            analysis = self.analyze_bullet(bullet, user_profile_data, match_analysis)
            analyses.append(analysis)
        
        return analyses
    
    def calculate_role_match_score(self, analyses: List[Dict]) -> float:
        """Calculate overall role match score (0-100)"""
        if not analyses:
            return 0.0
        
        # Weight scores by decision
        decision_weights = {
            "KEEP": 1.0,
            "REWRITE": 0.6,
            "DE_EMPHASIZE": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            weight = decision_weights.get(analysis["decision"], 0.5)
            bullet_score = (
                analysis["jd_similarity"] * 0.5 +
                analysis["keyword_score"] * 0.3 +
                (0.8 if analysis["has_evidence"] else 0.2) * 0.2
            )
            
            total_score += bullet_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        average_score = total_score / total_weight
        return round(average_score * 100, 2)

