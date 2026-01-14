"""
Profile Analyzer Module
Analyzes user profile to extract capabilities and match with job requirements
"""

import json
from typing import Dict, List, Tuple
from openai import OpenAI


class ProfileAnalyzer:
    """Analyzes user profile to understand capabilities and match with job requirements"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI API key"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = OpenAI(api_key=api_key)
        
        self.model = model
    
    def analyze_profile_capabilities(self, user_profile: Dict) -> Dict:
        """
        Analyze user profile to extract key capabilities, skills, and experiences
        Returns structured analysis of what the user has
        """
        system_prompt = """You are an expert at analyzing professional profiles. Extract and structure key capabilities from a user's profile.

Analyze the profile data (GitHub repos, LinkedIn experience, publications, etc.) and extract:
- core_skills: Technical skills and programming languages the user is proficient in
- technologies: Frameworks, tools, platforms they've worked with
- experiences: Key work experiences and achievements
- projects: Notable projects and their technologies
- achievements: Quantifiable achievements, metrics, impact
- domain_expertise: Areas of expertise (e.g., "Machine Learning", "Full-stack Development")
- education_background: Educational qualifications if available

Return ONLY valid JSON, no other text."""

        # Prepare profile summary for analysis
        profile_summary = self._prepare_profile_summary(user_profile)
        
        user_prompt = f"""Analyze this professional profile and extract structured capabilities:

{profile_summary}

Return a JSON object with these keys:
{{
    "core_skills": ["skill1", "skill2", ...],
    "technologies": ["tech1", "tech2", ...],
    "experiences": ["experience1", "experience2", ...],
    "projects": ["project1", "project2", ...],
    "achievements": ["achievement1 with metrics", ...],
    "domain_expertise": ["domain1", "domain2", ...],
    "education_background": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            
            capabilities = {
                "core_skills": result.get("core_skills", []),
                "technologies": result.get("technologies", []),
                "experiences": result.get("experiences", []),
                "projects": result.get("projects", []),
                "achievements": result.get("achievements", []),
                "domain_expertise": result.get("domain_expertise", []),
                "education_background": result.get("education_background", "Not specified")
            }
            
            print(f"✓ Profile analyzed: {len(capabilities['core_skills'])} skills, {len(capabilities['projects'])} projects")
            
            return capabilities
            
        except Exception as e:
            print(f"Error analyzing profile: {e}")
            return self._fallback_analysis(user_profile)
    
    def match_profile_with_job(self, profile_capabilities: Dict, job_description: Dict) -> Dict:
        """
        Match profile capabilities with job requirements
        Returns analysis of matches, gaps, and recommendations
        """
        system_prompt = """You are an expert at matching candidate profiles with job requirements. 
Analyze what the candidate has vs what the job requires, and provide actionable recommendations.

Compare:
1. Skills match: What skills from the job does the candidate have?
2. Missing skills: What required skills are missing?
3. Strengths: What does the candidate excel at that's relevant?
4. Recommendations: What should be emphasized/added to the resume?
5. Evidence: Specific examples from profile that support each skill/experience

IMPORTANT: Only recommend additions that have supporting evidence in the profile. Never invent skills or experiences.

Return ONLY valid JSON, no other text."""

        user_prompt = f"""Match this candidate profile with the job requirements:

CANDIDATE PROFILE CAPABILITIES:
Core Skills: {', '.join(profile_capabilities.get('core_skills', []))}
Technologies: {', '.join(profile_capabilities.get('technologies', []))}
Domain Expertise: {', '.join(profile_capabilities.get('domain_expertise', []))}
Key Experiences: {profile_capabilities.get('experiences', [])}
Projects: {profile_capabilities.get('projects', [])}
Achievements: {profile_capabilities.get('achievements', [])}

JOB REQUIREMENTS:
Role: {job_description.get('role', 'N/A')}
Required Skills: {', '.join(job_description.get('skills', []))}
Responsibilities: {job_description.get('responsibilities', [])}
Requirements: {job_description.get('requirements', [])}

Analyze and return JSON with:
{{
    "skill_matches": {{"skill": "evidence from profile", ...}},
    "missing_skills": ["skill1", "skill2", ...],
    "strengths": ["strength1", "strength2", ...],
    "recommendations": [
        {{
            "action": "EMPHASIZE" or "ADD" or "REWRITE",
            "skill_or_topic": "...",
            "evidence": "specific example from profile",
            "suggestion": "how to phrase it in resume"
        }}
    ],
    "match_score": 0-100
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            match_analysis = {
                "skill_matches": result.get("skill_matches", {}),
                "missing_skills": result.get("missing_skills", []),
                "strengths": result.get("strengths", []),
                "recommendations": result.get("recommendations", []),
                "match_score": result.get("match_score", 0)
            }
            
            print(f"✓ Profile matched with job: {match_analysis['match_score']}% match")
            print(f"  Skill matches: {len(match_analysis['skill_matches'])}")
            print(f"  Missing skills: {len(match_analysis['missing_skills'])}")
            print(f"  Recommendations: {len(match_analysis['recommendations'])}")
            
            return match_analysis
            
        except Exception as e:
            print(f"Error matching profile: {e}")
            return {
                "skill_matches": {},
                "missing_skills": [],
                "strengths": [],
                "recommendations": [],
                "match_score": 0
            }
    
    def get_recommendations_for_bullet(self, bullet: Dict, match_analysis: Dict, 
                                       profile_capabilities: Dict) -> Dict:
        """
        Get specific recommendations for a resume bullet based on profile analysis
        """
        bullet_text = bullet["text"]
        
        # Check if bullet should be enhanced based on recommendations
        relevant_recommendations = []
        for rec in match_analysis.get("recommendations", []):
            action = rec.get("action", "")
            skill_or_topic = rec.get("skill_or_topic", "").lower()
            
            # Check if recommendation is relevant to this bullet
            if skill_or_topic and (skill_or_topic in bullet_text.lower() or 
                                  any(word in bullet_text.lower() for word in skill_or_topic.split())):
                relevant_recommendations.append(rec)
        
        # Check if bullet mentions a strength that should be emphasized
        relevant_strengths = []
        for strength in match_analysis.get("strengths", []):
            if any(word in bullet_text.lower() for word in strength.lower().split()[:3]):
                relevant_strengths.append(strength)
        
        return {
            "bullet": bullet,
            "relevant_recommendations": relevant_recommendations,
            "relevant_strengths": relevant_strengths,
            "should_enhance": len(relevant_recommendations) > 0 or len(relevant_strengths) > 0,
            "enhancement_evidence": [rec.get("evidence", "") for rec in relevant_recommendations]
        }
    
    def _prepare_profile_summary(self, user_profile: Dict) -> str:
        """Prepare a summary of user profile for AI analysis"""
        summary_parts = []
        
        # GitHub data
        if "github" in user_profile and "repositories" in user_profile["github"]:
            repos = user_profile["github"]["repositories"]
            summary_parts.append("GITHUB REPOSITORIES:")
            for repo in repos[:10]:  # Top 10 repos
                repo_info = f"- {repo.get('name', 'Unknown')}: {repo.get('description', '')}"
                if repo.get('language'):
                    repo_info += f" (Language: {repo.get('language')})"
                if repo.get('key_bullets'):
                    repo_info += f" Key points: {', '.join(repo.get('key_bullets', [])[:3])}"
                summary_parts.append(repo_info)
            
            # Languages
            if "languages" in user_profile["github"]:
                langs = list(user_profile["github"]["languages"].keys())[:15]
                summary_parts.append(f"\nProgramming Languages: {', '.join(langs)}")
        
        # LinkedIn data (if available)
        if "linkedin" in user_profile:
            linkedin = user_profile["linkedin"]
            if "experience" in linkedin and linkedin["experience"]:
                summary_parts.append("\nWORK EXPERIENCE:")
                for exp in linkedin["experience"][:5]:
                    exp_text = f"- {exp.get('title', '')} at {exp.get('company', '')}"
                    if exp.get('description'):
                        exp_text += f": {exp.get('description')}"
                    summary_parts.append(exp_text)
        
        # Google Scholar data
        if "scholar" in user_profile and "publications" in user_profile["scholar"]:
            pubs = user_profile["scholar"]["publications"]
            summary_parts.append("\nPUBLICATIONS:")
            for pub in pubs[:5]:
                pub_text = f"- {pub.get('title', '')} ({pub.get('venue', '')})"
                if pub.get('citations', 0) > 0:
                    pub_text += f" - {pub.get('citations')} citations"
                summary_parts.append(pub_text)
        
        # Bullets from profile
        if "bullets" in user_profile:
            summary_parts.append("\nPROFILE SUMMARY BULLETS:")
            for bullet in user_profile["bullets"][:20]:
                summary_parts.append(f"- {bullet}")
        
        return "\n".join(summary_parts)
    
    def _fallback_analysis(self, user_profile: Dict) -> Dict:
        """Fallback analysis if AI fails"""
        capabilities = {
            "core_skills": [],
            "technologies": [],
            "experiences": [],
            "projects": [],
            "achievements": [],
            "domain_expertise": [],
            "education_background": "Not specified"
        }
        
        # Extract from profile bullets
        if "bullets" in user_profile:
            capabilities["experiences"] = user_profile["bullets"][:10]
        
        # Extract languages from GitHub
        if "github" in user_profile and "languages" in user_profile["github"]:
            capabilities["core_skills"] = list(user_profile["github"]["languages"].keys())[:20]
        
        return capabilities

