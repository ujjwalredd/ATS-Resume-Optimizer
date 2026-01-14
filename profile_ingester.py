"""
User Profile Ingestion Module
Fetches and parses GitHub, LinkedIn, and Google Scholar profiles
"""

import json
import re
import requests
from typing import List, Dict, Optional
from github import Github
from scholarly import scholarly
from bs4 import BeautifulSoup
import time


class ProfileIngester:
    """Ingests user profile data from multiple sources"""
    
    def __init__(self, github_token: str, github_username: str):
        self.github_token = github_token
        self.github_username = github_username
        self.github = Github(github_token) if github_token else None
        
    def ingest_github_profile(self) -> Dict:
        """Fetch GitHub repositories, READMEs, languages, and commits"""
        if not self.github:
            return {"error": "GitHub token not provided"}
            
        profile_data = {
            "repositories": [],
            "languages": {},
            "total_commits": 0,
            "repos_count": 0
        }
        
        try:
            user = self.github.get_user(self.github_username)
            repos = user.get_repos()
            
            for repo in repos:
                if repo.archived or repo.fork:
                    continue
                    
                repo_data = {
                    "name": repo.name,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "topics": repo.get_topics(),
                    "readme_content": "",
                    "key_bullets": []
                }
                
                # Fetch README
                try:
                    readme = repo.get_readme()
                    readme_content = readme.decoded_content.decode('utf-8')
                    repo_data["readme_content"] = readme_content
                    repo_data["key_bullets"] = self._extract_bullets_from_readme(readme_content)
                except:
                    pass
                
                # Aggregate languages
                if repo.language:
                    profile_data["languages"][repo.language] = \
                        profile_data["languages"].get(repo.language, 0) + 1
                
                # Get commit count (approximate)
                try:
                    commits = repo.get_commits(author=user.login)
                    commit_count = sum(1 for _ in commits[:100])  # Limit to avoid rate limits
                    repo_data["commits"] = commit_count
                    profile_data["total_commits"] += commit_count
                except:
                    repo_data["commits"] = 0
                
                profile_data["repositories"].append(repo_data)
                profile_data["repos_count"] += 1
                
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            return {"error": f"GitHub ingestion failed: {str(e)}"}
        
        return profile_data
    
    def _extract_bullets_from_readme(self, readme_content: str) -> List[str]:
        """Extract bullet points and key features from README"""
        bullets = []
        
        # Extract markdown bullets
        bullet_pattern = r'^[-*+]\s+(.+)$'
        for line in readme_content.split('\n'):
            match = re.match(bullet_pattern, line.strip())
            if match:
                bullets.append(match.group(1).strip())
        
        # Extract numbered lists
        numbered_pattern = r'^\d+[\.)]\s+(.+)$'
        for line in readme_content.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                bullets.append(match.group(1).strip())
        
        return bullets[:10]  # Limit to top 10
    
    def ingest_linkedin_profile(self, profile_url: Optional[str] = None) -> Dict:
        """
        Parse LinkedIn profile data
        Note: LinkedIn API requires special access. This uses web scraping as fallback.
        For production, consider using official LinkedIn API or scraping services.
        """
        profile_data = {
            "experience": [],
            "education": [],
            "skills": [],
            "summary": ""
        }
        
        # TODO: Implement LinkedIn scraping with Selenium or use official API
        # For now, return structure for manual input or API integration
        return {
            "message": "LinkedIn ingestion requires manual implementation or API access",
            "structure": profile_data
        }
    
    def ingest_google_scholar(self, scholar_id: Optional[str] = None, author_name: Optional[str] = None) -> Dict:
        """Fetch Google Scholar publications"""
        publications_data = {
            "publications": [],
            "total_citations": 0,
            "h_index": 0
        }
        
        try:
            if scholar_id:
                search_query = scholarly.search_author_id(scholar_id)
            elif author_name:
                search_query = scholarly.search_author(author_name)
                search_query = next(search_query)
            else:
                return {"error": "Either scholar_id or author_name required"}
            
            author = scholarly.fill(search_query)
            
            publications_data["total_citations"] = author.get("citedby", 0)
            publications_data["h_index"] = author.get("hindex", 0)
            
            # Get publications
            for pub in author.get("publications", [])[:20]:  # Limit to recent 20
                filled_pub = scholarly.fill(pub)
                pub_data = {
                    "title": filled_pub.get("bib", {}).get("title", ""),
                    "authors": filled_pub.get("bib", {}).get("author", []),
                    "venue": filled_pub.get("bib", {}).get("venue", ""),
                    "year": filled_pub.get("bib", {}).get("pub_year", ""),
                    "citations": filled_pub.get("num_citations", 0),
                    "abstract": filled_pub.get("bib", {}).get("abstract", ""),
                    "url": filled_pub.get("pub_url", "")
                }
                publications_data["publications"].append(pub_data)
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            return {"error": f"Google Scholar ingestion failed: {str(e)}"}
        
        return publications_data
    
    def convert_to_bullets(self, profile_data: Dict) -> List[str]:
        """Convert all profile data into bullet-like text for embedding"""
        bullets = []
        
        # GitHub bullets
        if "repositories" in profile_data:
            for repo in profile_data["repositories"]:
                if repo.get("description"):
                    bullets.append(f"Built {repo['name']}: {repo['description']}")
                bullets.extend(repo.get("key_bullets", []))
                if repo.get("language"):
                    bullets.append(f"Developed {repo['name']} in {repo['language']}")
        
        # Skills from languages
        if "languages" in profile_data:
            lang_list = ", ".join(sorted(profile_data["languages"].keys()))
            bullets.append(f"Proficient in programming languages: {lang_list}")
        
        # LinkedIn experience
        if "experience" in profile_data:
            for exp in profile_data["experience"]:
                bullet = f"{exp.get('title', '')} at {exp.get('company', '')}"
                if exp.get("description"):
                    bullets.append(bullet + f": {exp['description']}")
                else:
                    bullets.append(bullet)
        
        # Education
        if "education" in profile_data:
            for edu in profile_data["education"]:
                bullets.append(f"{edu.get('degree', '')} from {edu.get('school', '')}")
        
        # Publications
        if "publications" in profile_data:
            for pub in profile_data["publications"]:
                pub_text = f"Published '{pub.get('title', '')}' in {pub.get('venue', '')}"
                if pub.get("citations", 0) > 0:
                    pub_text += f" ({pub['citations']} citations)"
                bullets.append(pub_text)
                if pub.get("abstract"):
                    bullets.append(f"Research focus: {pub['abstract'][:200]}")
        
        return bullets
    
    def ingest_all(self, scholar_id: Optional[str] = None, author_name: Optional[str] = None) -> Dict:
        """Ingest all profile data sources"""
        combined_data = {
            "github": {},
            "linkedin": {},
            "scholar": {},
            "bullets": []
        }
        
        # GitHub
        print("Ingesting GitHub profile...")
        combined_data["github"] = self.ingest_github_profile()
        
        # LinkedIn (placeholder)
        print("Ingesting LinkedIn profile...")
        combined_data["linkedin"] = self.ingest_linkedin_profile()
        
        # Google Scholar
        if scholar_id or author_name:
            print("Ingesting Google Scholar profile...")
            combined_data["scholar"] = self.ingest_google_scholar(scholar_id, author_name)
        
        # Convert to bullets
        print("Converting to bullet format...")
        combined_data["bullets"] = self.convert_to_bullets(combined_data)
        
        return combined_data

