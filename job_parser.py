"""
Job Description Parser Module
Uses OpenAI API to parse and structure job descriptions
Can extract from URLs or parse text directly
"""

import json
import re
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from openai import OpenAI


class JobParser:
    """Parses job descriptions into structured format using AI"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """Initialize with OpenAI API key"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Provide via api_key parameter or OPENAI_API_KEY env var")
            self.client = OpenAI(api_key=api_key)
        
        self.model = model
    
    def parse_job_description(self, jd_input: str) -> Dict:
        """
        Parse job description from URL or text
        Accepts either a URL or raw text
        """
        # Check if input is a URL
        if self._is_url(jd_input):
            print(f"Detected URL: {jd_input}")
            jd_text = self._extract_from_url(jd_input)
            if not jd_text:
                raise ValueError(f"Failed to extract job description from URL: {jd_input}")
        else:
            jd_text = jd_input
        
        return self._parse_jd_text(jd_text, jd_input if self._is_url(jd_input) else None)
    
    def _is_url(self, text: str) -> bool:
        """Check if input is a URL"""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_from_url(self, url: str) -> str:
        """Extract job description text from URL"""
        try:
            # Set headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            print(f"Fetching content from URL...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try different extraction strategies based on common job board patterns
            jd_text = self._extract_jd_by_platform(url, soup)
            
            if not jd_text or len(jd_text) < 100:
                # Fallback: extract all text from main content
                jd_text = self._extract_all_text(soup)
            
            if jd_text and len(jd_text) > 100:
                print(f"✓ Extracted {len(jd_text)} characters from URL")
                return jd_text
            else:
                print("Warning: Extracted text seems too short, using full page content")
                return self._extract_all_text(soup)
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return ""
        except Exception as e:
            print(f"Error extracting from URL: {e}")
            return ""
    
    def _extract_jd_by_platform(self, url: str, soup: BeautifulSoup) -> str:
        """Extract job description based on platform-specific selectors"""
        url_lower = url.lower()
        jd_text = ""
        
        # LinkedIn
        if 'linkedin.com/jobs' in url_lower:
            # Try common LinkedIn job description selectors
            selectors = [
                'div[class*="description__text"]',
                'div.show-more-less-html__markup',
                'section.jobs-description__content',
                'div[data-automation-id="jobPostingDescription"]'
            ]
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    jd_text = element.get_text(separator='\n', strip=True)
                    if len(jd_text) > 200:
                        break
        
        # Indeed
        elif 'indeed.com' in url_lower:
            selectors = [
                'div[id="jobDescriptionText"]',
                'div[data-testid="job-description"]',
                'div.jobsearch-jobDescriptionText'
            ]
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    jd_text = element.get_text(separator='\n', strip=True)
                    if len(jd_text) > 200:
                        break
        
        # Glassdoor
        elif 'glassdoor.com' in url_lower:
            selectors = [
                'div[data-test="jobDescriptionText"]',
                'div.jobDescriptionContent',
                'div[class*="jobDescription"]'
            ]
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    jd_text = element.get_text(separator='\n', strip=True)
                    if len(jd_text) > 200:
                        break
        
        # Generic: Try common job description class/id patterns
        if not jd_text or len(jd_text) < 200:
            generic_selectors = [
                'div[class*="description"]',
                'div[class*="job-description"]',
                'div[id*="description"]',
                'article',
                'main',
                'div[role="main"]'
            ]
            for selector in generic_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator='\n', strip=True)
                    # Look for job-like content (contains keywords)
                    if len(text) > 300 and any(keyword in text.lower() for keyword in 
                                               ['responsibilities', 'requirements', 'qualifications', 'job', 'position']):
                        jd_text = text
                        break
                if jd_text:
                    break
        
        return jd_text
    
    def _extract_all_text(self, soup: BeautifulSoup) -> str:
        """Extract all text content from page as fallback"""
        # Get text from body
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        return soup.get_text(separator='\n', strip=True)
    
    def _parse_jd_text(self, jd_text: str, source_url: Optional[str] = None) -> Dict:
        """Parse raw job description text into structured JSON using AI"""
        
        system_prompt = """You are an expert at parsing job descriptions. Extract structured information from job postings and return valid JSON.

Extract the following fields:
- role: Job title/position (e.g., "Senior Software Engineer")
- company: Company name (or "Not specified" if not mentioned)
- location: Job location (e.g., "San Francisco, CA" or "Remote")
- skills: List of technical skills, programming languages, tools, frameworks mentioned
- responsibilities: List of job responsibilities and duties (as separate items)
- requirements: List of job requirements, qualifications, must-haves (as separate items)
- keywords: Important keywords and phrases relevant to the role (combine skills, technologies, domain terms)
- experience_level: One of "Entry-level", "Mid-level", "Senior", "Executive", or "Not specified"
- education: Education requirement like "Bachelor's", "Master's", "PhD", or "Not specified"

Return ONLY valid JSON, no other text."""

        user_prompt = f"""Parse this job description and extract structured information:

{jd_text}

Return the result as a JSON object with these exact keys:
{{
    "role": "...",
    "company": "...",
    "location": "...",
    "skills": ["skill1", "skill2", ...],
    "responsibilities": ["responsibility1", "responsibility2", ...],
    "requirements": ["requirement1", "requirement2", ...],
    "keywords": ["keyword1", "keyword2", ...],
    "experience_level": "...",
    "education": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},  # Force JSON output
                temperature=0.1  # Low temperature for consistent parsing
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields exist
            structured_jd = {
                "role": result.get("role", "Unknown Role"),
                "company": result.get("company", "Unknown Company"),
                "location": result.get("location", "Not specified"),
                "skills": result.get("skills", []),
                "responsibilities": result.get("responsibilities", []),
                "requirements": result.get("requirements", []),
                "keywords": result.get("keywords", []),
                "experience_level": result.get("experience_level", "Not specified"),
                "education": result.get("education", "Not specified"),
                "raw_text": jd_text,
                "source_url": source_url if source_url else None
            }
            
            # Limit array sizes to prevent too many items
            structured_jd["skills"] = structured_jd["skills"][:50]
            structured_jd["responsibilities"] = structured_jd["responsibilities"][:30]
            structured_jd["requirements"] = structured_jd["requirements"][:30]
            structured_jd["keywords"] = structured_jd["keywords"][:50]
            
            print(f"✓ Parsed job description: {structured_jd['role']} at {structured_jd['company']}")
            print(f"  Skills: {len(structured_jd['skills'])}")
            print(f"  Responsibilities: {len(structured_jd['responsibilities'])}")
            print(f"  Requirements: {len(structured_jd['requirements'])}")
            
            return structured_jd
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            try:
                print(f"Raw response: {response.choices[0].message.content}")
            except:
                pass
            # Fallback to basic extraction
            return self._fallback_parse(jd_text, source_url)
        except Exception as e:
            print(f"Error parsing job description: {e}")
            return self._fallback_parse(jd_text, source_url)
    
    def _fallback_parse(self, jd_text: str, source_url: Optional[str] = None) -> Dict:
        """Fallback parser if AI parsing fails"""
        return {
            "role": "Unknown Role",
            "company": "Unknown Company",
            "location": "Not specified",
            "skills": [],
            "responsibilities": [],
            "requirements": [],
            "keywords": [],
            "experience_level": "Not specified",
            "education": "Not specified",
            "raw_text": jd_text,
            "source_url": source_url
        }
    
    def get_keywords_for_embedding(self, jd_structured: Dict) -> str:
        """Combine structured data into a text for embedding"""
        keywords_text = f"{jd_structured['role']} "
        keywords_text += " ".join(jd_structured['skills']) + " "
        keywords_text += " ".join(jd_structured['keywords']) + " "
        keywords_text += " ".join(jd_structured['responsibilities'][:5]) + " "
        keywords_text += " ".join(jd_structured['requirements'][:5])
        
        return keywords_text
