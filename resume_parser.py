"""
Resume Parser Module
Uses OpenAI API to parse LaTeX resume files and extract bullets
"""

import json
import re
from typing import List, Dict
from pathlib import Path
from openai import OpenAI


class ResumeParser:
    """Parses LaTeX resume files using AI"""
    
    def __init__(self, tex_file_path: str, api_key: str = None, model: str = "gpt-4o-mini"):
        """Initialize parser with OpenAI API key"""
        self.tex_file_path = Path(tex_file_path)
        self.content = ""
        self.bullets = []
        self.bullet_positions = []  # (start_line, end_line, original_text)
        
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
        
    def load_resume(self):
        """Load LaTeX resume from file"""
        if not self.tex_file_path.exists():
            raise FileNotFoundError(f"Resume file not found: {self.tex_file_path}")
        
        with open(self.tex_file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        
        print(f"Loaded resume from {self.tex_file_path}")
    
    def extract_bullets(self) -> List[Dict]:
        """
        Extract all bullets from LaTeX resume using AI
        Returns list of bullet dictionaries with position and content
        """
        self.bullets = []
        self.bullet_positions = []
        
        system_prompt = """You are an expert at parsing LaTeX resumes. Extract all bullet points from the resume content.

For each bullet point found, extract:
- The bullet text content (cleaned of LaTeX formatting but preserving meaning)
- The original LaTeX code including the bullet marker (e.g., "\\item ..." or "- ...")
- The section/context where it appears (e.g., "Experience", "Projects", "Education")

Return a JSON array of bullet objects. Each bullet should have:
- text: The cleaned text content (human-readable, no LaTeX commands)
- original_latex: The original LaTeX code as it appears in the document
- section: The section name where this bullet appears (or "Unknown")
- index: A unique index number starting from 0

Return ONLY valid JSON, no other text."""

        # Limit content to avoid token limits
        content_sample = self.content[:8000] if len(self.content) > 8000 else self.content
        
        user_prompt = f"""Extract all bullet points from this LaTeX resume:

{content_sample}

Return as a JSON object with a "bullets" key containing an array of bullet objects. Each bullet should have:
- "text": cleaned text content
- "original_latex": original LaTeX code
- "section": section name
- "index": unique index

Format: {{"bullets": [{{"text": "...", "original_latex": "...", "section": "...", "index": 0}}, ...]}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},  # Force JSON output
                temperature=0.1
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Handle both array and object with array
            if isinstance(result, dict):
                bullets_data = result.get("bullets", result.get("items", result.get("bullet_points", [])))
            elif isinstance(result, list):
                bullets_data = result
            else:
                bullets_data = []
            
            # Convert to our format and find positions in original content
            lines = self.content.split('\n')
            
            for i, bullet_data in enumerate(bullets_data):
                bullet_text = bullet_data.get("text", "").strip()
                original_latex = bullet_data.get("original_latex", "")
                section = bullet_data.get("section", "Unknown")
                
                if len(bullet_text) < 10:  # Skip very short bullets
                    continue
                
                # Find position in original content
                start_pos = self.content.find(original_latex) if original_latex else -1
                end_pos = start_pos + len(original_latex) if start_pos >= 0 else -1
                
                if start_pos < 0:
                    # Fallback: try to find by text content
                    bullet_text_clean = self._clean_latex_for_search(bullet_text)
                    for j, line in enumerate(lines):
                        if bullet_text_clean.lower() in self._clean_latex_for_search(line).lower():
                            start_pos = sum(len(l) + 1 for l in lines[:j])
                            end_pos = start_pos + len(line)
                            start_line = j
                            end_line = j
                            break
                    else:
                        start_line = 0
                        end_line = 0
                else:
                    start_line = self.content[:start_pos].count('\n')
                    end_line = self.content[:end_pos].count('\n')
                
                bullet_dict = {
                    "text": bullet_text,
                    "original_latex": original_latex if original_latex else f"\\item {bullet_text}",
                    "start_line": start_line,
                    "end_line": end_line,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "type": "item" if "\\item" in original_latex.lower() else "text",
                    "section": section,
                    "index": i
                }
                
                self.bullets.append(bullet_dict)
                if start_pos >= 0:
                    self.bullet_positions.append((start_pos, end_pos, original_latex))
            
            # If AI parsing didn't work well, fall back to regex
            if len(self.bullets) == 0:
                print("  AI extraction found no bullets, falling back to regex parsing...")
                self._fallback_extract_bullets()
            
            print(f"✓ Extracted {len(self.bullets)} bullets from resume using AI")
            return self.bullets
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Falling back to regex parsing...")
            return self._fallback_extract_bullets()
        except Exception as e:
            print(f"Error extracting bullets with AI: {e}")
            print("Falling back to regex parsing...")
            return self._fallback_extract_bullets()
    
    def _fallback_extract_bullets(self) -> List[Dict]:
        """Fallback regex-based extraction if AI parsing fails"""
        self.bullets = []
        lines = self.content.split('\n')
        
        # Extract \item entries
        item_pattern = r'\\item\s*([^\n]+(?:\n(?!\\item|\\end)[^\n]+)*)'
        for match in re.finditer(item_pattern, self.content, re.MULTILINE):
            bullet_text = match.group(1).strip()
            bullet_text = self._clean_latex(bullet_text)
            
            if len(bullet_text) > 10:
                start_line = self.content[:match.start()].count('\n')
                end_line = self.content[:match.end()].count('\n')
                
                bullet_dict = {
                    "text": bullet_text,
                    "original_latex": match.group(0),
                    "start_line": start_line,
                    "end_line": end_line,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "type": "item",
                    "section": "Unknown",
                    "index": len(self.bullets)
                }
                
                self.bullets.append(bullet_dict)
        
        # Extract text bullets
        text_bullet_pattern = r'^\s*[-•]\s+(.+?)$'
        for i, line in enumerate(lines):
            match = re.match(text_bullet_pattern, line)
            if match:
                bullet_text = match.group(1).strip()
                bullet_text = self._clean_latex(bullet_text)
                
                if len(bullet_text) > 10 and not any(b["text"] == bullet_text for b in self.bullets):
                    bullet_dict = {
                        "text": bullet_text,
                        "original_latex": line,
                        "start_line": i,
                        "end_line": i,
                        "start_pos": sum(len(l) + 1 for l in lines[:i]),
                        "end_pos": sum(len(l) + 1 for l in lines[:i+1]),
                        "type": "text",
                        "section": "Unknown",
                        "index": len(self.bullets)
                    }
                    self.bullets.append(bullet_dict)
        
        print(f"  Extracted {len(self.bullets)} bullets using regex fallback")
        return self.bullets
    
    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands while preserving content"""
        # Remove LaTeX commands like \textbf{}, \textit{}, etc.
        text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)
        
        # Remove standalone commands
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        
        # Remove braces
        text = re.sub(r'[{}]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _clean_latex_for_search(self, text: str) -> str:
        """Clean LaTeX for searching (less aggressive than _clean_latex)"""
        # Only remove formatting commands, keep structure
        text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]+)\}', r'\1', text)
        return text.strip()
    
    def replace_bullet(self, old_bullet: Dict, new_text: str):
        """
        Replace a bullet with new text while preserving LaTeX formatting
        """
        old_latex = old_bullet["original_latex"]
        
        # Preserve LaTeX structure
        if old_bullet["type"] == "item":
            # Maintain \item structure
            new_latex = f"\\item {new_text}"
        else:
            # Maintain text bullet structure
            new_latex = f"- {new_text}"
        
        # Replace in content
        start = old_bullet["start_pos"]
        end = old_bullet["end_pos"]
        
        if start >= 0 and end >= 0:
            # Find the exact match in content
            content_slice = self.content[start:end]
            if content_slice.strip() == old_latex.strip() or old_latex in content_slice:
                self.content = self.content[:start] + new_latex + self.content[end:]
            else:
                # Try finding by text content
                self.content = self.content.replace(old_latex, new_latex, 1)
        else:
            # Fallback: replace by text matching
            self.content = self.content.replace(old_latex, new_latex, 1)
        
        # Update bullet dictionary
        old_bullet["text"] = new_text
        old_bullet["original_latex"] = new_latex
    
    def remove_bullet(self, bullet: Dict):
        """Remove a bullet from the resume"""
        old_latex = bullet["original_latex"]
        start = bullet.get("start_pos", -1)
        end = bullet.get("end_pos", -1)
        
        if start >= 0 and end >= 0:
            self.content = self.content[:start] + self.content[end:].lstrip()
        else:
            self.content = self.content.replace(old_latex, "", 1).strip()
        
        if bullet in self.bullets:
            self.bullets.remove(bullet)
    
    def save_resume(self, output_path: str = None):
        """Save modified resume to file"""
        if output_path is None:
            output_path = self.tex_file_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.content)
        
        print(f"Saved resume to {output_path}")
    
    def get_section_bullets(self, section_keywords: List[str] = None) -> List[Dict]:
        """Get bullets from specific sections"""
        if section_keywords is None:
            section_keywords = ["experience", "work", "project", "education"]
        
        if not self.bullets:
            return []
        
        section_bullets = []
        for bullet in self.bullets:
            section = bullet.get("section", "Unknown").lower()
            if any(keyword.lower() in section for keyword in section_keywords):
                section_bullets.append(bullet)
        
        return section_bullets if section_bullets else self.bullets
