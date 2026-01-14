"""
Rewrite Engine Module
Rewrites resume bullets using OpenAI API with profile context
"""

import json
from typing import List, Dict, Optional
from openai import OpenAI


class RewriteEngine:
    """Rewrites resume bullets using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", 
                 temperature: float = 0.3, max_tokens: int = 500):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def rewrite_bullet(self, bullet: Dict, job_keywords: str, 
                      relevant_profile_entries: List[Dict],
                      original_bullet_context: str = "",
                      match_analysis: Dict = None,
                      profile_capabilities: Dict = None) -> str:
        """
        Rewrite a single bullet point with job context and profile evidence
        Enhanced with profile match analysis
        """
        
        # Prepare profile context
        profile_context = self._prepare_profile_context(relevant_profile_entries)
        
        # Add recommendations and strengths from match analysis
        enhancement_guidance = ""
        if match_analysis:
            strengths = match_analysis.get("strengths", [])
            if strengths:
                enhancement_guidance += f"\n\nPROFILE STRENGTHS TO EMPHASIZE: {', '.join(strengths[:5])}"
            
            # Find relevant recommendations for this bullet
            for rec in match_analysis.get("recommendations", []):
                skill_or_topic = rec.get("skill_or_topic", "").lower()
                if skill_or_topic and skill_or_topic in bullet["text"].lower():
                    enhancement_guidance += f"\n\nRECOMMENDATION: {rec.get('suggestion', '')}"
                    enhancement_guidance += f"\nEVIDENCE: {rec.get('evidence', '')}"
        
        # Create prompt
        prompt = self._create_rewrite_prompt(
            bullet_text=bullet["text"],
            job_keywords=job_keywords,
            profile_context=profile_context,
            context=original_bullet_context,
            enhancement_guidance=enhancement_guidance
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # Clean up LaTeX unsafe characters if needed
            rewritten = self._sanitize_latex(rewritten)
            
            return rewritten
            
        except Exception as e:
            print(f"Error rewriting bullet: {e}")
            return bullet["text"]  # Return original on error
    
    def _get_system_prompt(self) -> str:
        """System prompt for OpenAI"""
        return """You are an expert resume writer specializing in ATS (Applicant Tracking System) optimization. 
Your task is to rewrite resume bullet points to:
1. Include relevant keywords from the job description naturally
2. Incorporate specific achievements and metrics from the user's profile
3. Use strong action verbs and quantifiable results
4. Maintain factual accuracy - never invent experiences or skills
5. Ensure the text is clean and professional (no LaTeX commands, no special characters that break LaTeX)

Return ONLY the rewritten bullet text, nothing else. Keep it concise (one line, under 200 characters if possible)."""
    
    def _create_rewrite_prompt(self, bullet_text: str, job_keywords: str,
                              profile_context: str, context: str = "",
                              enhancement_guidance: str = "") -> str:
        """Create the rewrite prompt with profile analysis guidance"""
        prompt = f"""Rewrite this resume bullet point to better match the job description while incorporating relevant profile information.

ORIGINAL BULLET:
{bullet_text}

JOB DESCRIPTION KEYWORDS AND REQUIREMENTS:
{job_keywords[:1000]}

RELEVANT PROFILE INFORMATION (use as evidence, but maintain accuracy):
{profile_context[:800]}

{enhancement_guidance}

{f'ADDITIONAL CONTEXT: {context[:200]}' if context else ''}

INSTRUCTIONS:
- Incorporate relevant keywords from the job description naturally
- Use specific details from the profile information when available
- Emphasize profile strengths that match job requirements
- Include metrics and achievements (numbers, percentages, scale)
- Use strong action verbs (e.g., "Developed", "Implemented", "Led", "Optimized")
- Maintain factual accuracy - ONLY use information from the profile provided above
- Never invent skills, experiences, or metrics that aren't in the profile
- Keep it concise and impactful (one line, under 200 characters)
- Do NOT include LaTeX formatting or special characters

Rewritten bullet:"""
        
        return prompt
    
    def _prepare_profile_context(self, relevant_entries: List[Dict]) -> str:
        """Prepare profile entries as context text"""
        if not relevant_entries:
            return "No specific profile context available."
        
        context_lines = []
        for entry in relevant_entries[:5]:  # Top 5 most relevant
            if isinstance(entry, dict):
                text = entry.get("text", "")
                if text:
                    context_lines.append(f"- {text}")
        
        return "\n".join(context_lines) if context_lines else "No specific profile context available."
    
    def _sanitize_latex(self, text: str) -> str:
        """Remove or escape LaTeX-unsafe characters"""
        # Remove LaTeX commands that might have been generated
        text = text.replace('\\textbf{', '').replace('\\textit{', '').replace('\\emph{', '')
        text = text.replace('{', '').replace('}', '')
        
        # Remove special LaTeX characters that could break compilation
        # But keep common punctuation
        text = text.replace('&', 'and')
        text = text.replace('%', 'percent')
        text = text.replace('$', '')
        text = text.replace('#', '')
        text = text.replace('^', '')
        text = text.replace('_', ' ')  # Replace underscore with space
        
        # Clean up multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_cover_letter(self, job_description: Dict, user_profile: Dict, 
                            role_match_score: float) -> str:
        """Generate a cover letter based on job description and profile"""
        
        prompt = f"""Write a professional cover letter for this job application.

JOB DESCRIPTION:
Role: {job_description.get('role', 'N/A')}
Company: {job_description.get('company', 'N/A')}
Key Requirements: {', '.join(job_description.get('requirements', [])[:5])}

CANDIDATE PROFILE HIGHLIGHTS:
{self._summarize_profile(user_profile)}

ROLE MATCH SCORE: {role_match_score}% (based on resume analysis)

INSTRUCTIONS:
- Write a concise, professional cover letter (3-4 paragraphs)
- Highlight relevant experience and skills that match the job
- Show enthusiasm for the role and company
- Include specific achievements from the profile
- Maintain a professional but engaging tone
- Do NOT invent experiences

Cover Letter:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional cover letter writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating cover letter: {e}")
            return "Error generating cover letter. Please try again."
    
    def generate_recruiter_message(self, job_description: Dict, role_match_score: float) -> str:
        """Generate a brief LinkedIn message for recruiters"""
        
        prompt = f"""Write a brief, professional LinkedIn message to a recruiter for this position.

JOB: {job_description.get('role', 'N/A')} at {job_description.get('company', 'N/A')}
ROLE MATCH SCORE: {role_match_score}%

INSTRUCTIONS:
- Keep it concise (2-3 sentences)
- Express interest in the role
- Mention one key qualification or achievement
- Professional and friendly tone
- Include a call to action

LinkedIn Message:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional networker writing LinkedIn messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating recruiter message: {e}")
            return "Error generating message. Please try again."
    
    def _summarize_profile(self, profile: Dict) -> str:
        """Create a brief summary of user profile for cover letter"""
        summary_parts = []
        
        if "github" in profile and "repositories" in profile["github"]:
            repo_count = len(profile["github"].get("repositories", []))
            if repo_count > 0:
                summary_parts.append(f"- {repo_count} GitHub repositories")
        
        if "scholar" in profile and "publications" in profile["scholar"]:
            pub_count = len(profile["scholar"].get("publications", []))
            if pub_count > 0:
                summary_parts.append(f"- {pub_count} published research papers")
        
        return "\n".join(summary_parts) if summary_parts else "Profile information available."

