"""
Main Workflow Orchestrator
Coordinates all modules for ATS resume optimization
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

from profile_ingester import ProfileIngester
from embedding_store import EmbeddingStore
from job_parser import JobParser
from resume_parser import ResumeParser
from alignment_engine import AlignmentEngine
from rewrite_engine import RewriteEngine
from github_integration import GitHubIntegration
from profile_analyzer import ProfileAnalyzer


class ATSResumeOptimizer:
    """Main orchestrator for ATS resume optimization workflow"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        load_dotenv()
        
        self.profile_ingester = ProfileIngester(
            github_token=self.config.get("github", {}).get("token") or os.getenv("GITHUB_TOKEN"),
            github_username=self.config.get("github", {}).get("username") or os.getenv("GITHUB_USERNAME")
        )
        
        self.embedding_store = EmbeddingStore(
            model_name=self.config.get("embeddings", {}).get("model", "all-MiniLM-L6-v2"),
            db_path=self.config.get("embeddings", {}).get("vector_db_path", "embeddings_db")
        )
        
        openai_api_key = self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        parsing_model = self.config.get("openai_settings", {}).get("parsing_model") or \
                       self.config.get("openai_settings", {}).get("model", "gpt-4o-mini")
        
        self.job_parser = JobParser(
            api_key=openai_api_key,
            model=parsing_model
        )
        self.alignment_engine = AlignmentEngine(
            embedding_store=self.embedding_store,
            similarity_threshold=self.config.get("analysis", {}).get("similarity_threshold", 0.6),
            rewrite_threshold=self.config.get("analysis", {}).get("rewrite_threshold", 0.4),
            keep_threshold=self.config.get("analysis", {}).get("keep_threshold", 0.75)
        )
        
        self.rewrite_engine = RewriteEngine(
            api_key=self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY"),
            model=self.config.get("openai_settings", {}).get("model", "gpt-4-turbo-preview"),
            temperature=self.config.get("openai_settings", {}).get("temperature", 0.3),
            max_tokens=self.config.get("openai_settings", {}).get("max_tokens", 500)
        )
        
        self.github_integration = GitHubIntegration(
            token=self.config.get("github", {}).get("token") or os.getenv("GITHUB_TOKEN"),
            repo_owner=self.config.get("repository", {}).get("owner"),
            repo_name=self.config.get("repository", {}).get("name"),
            branch=self.config.get("repository", {}).get("branch", "main")
        )
        
        self.profile_analyzer = ProfileAnalyzer(
            api_key=openai_api_key,
            model=parsing_model
        )
        
        self.user_profile = None
        self.profile_capabilities = None
        self.match_analysis = None
        self.job_description = None
        self.resume_parser = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def step1_ingest_profile(self, scholar_id: Optional[str] = None, 
                            author_name: Optional[str] = None) -> Dict:
        """Step 1: Ingest user profile from all sources"""
        print("\n=== STEP 1: Ingesting User Profile ===")
        
        self.user_profile = self.profile_ingester.ingest_all(
            scholar_id=scholar_id,
            author_name=author_name
        )
        
        print(f"✓ Profile ingested: {len(self.user_profile.get('bullets', []))} bullets extracted")
        return self.user_profile
    
    def step2_create_embedding_store(self, rebuild: bool = False) -> EmbeddingStore:
        """Step 2: Create embedding store for user data"""
        print("\n=== STEP 2: Creating Embedding Store ===")
        
        embedding_path = os.path.join(
            self.config.get("embeddings", {}).get("vector_db_path", "embeddings_db"),
            "index"
        )
        
        if not rebuild and os.path.exists(f"{embedding_path}.faiss"):
            try:
                self.embedding_store.load(embedding_path)
                print("✓ Loaded existing embedding store")
                return self.embedding_store
            except Exception as e:
                print(f"Error loading store: {e}. Rebuilding...")
        
        bullets = self.user_profile.get("bullets", [])
        if not bullets:
            raise ValueError("No bullets found in user profile. Run step1_ingest_profile first.")
        
        embeddings = self.embedding_store.create_embeddings(
            texts=bullets,
            metadata=[{"text": bullet, "source": "profile", "index": i} 
                     for i, bullet in enumerate(bullets)]
        )
        
        self.embedding_store.build_index(
            embeddings=embeddings,
            texts=bullets,
            metadata=[{"text": bullet, "source": "profile", "index": i} 
                     for i, bullet in enumerate(bullets)]
        )
        
        self.embedding_store.save(embedding_path)
        print(f"✓ Created embedding store with {len(bullets)} entries")
        
        return self.embedding_store
    
    def step2b_analyze_profile(self) -> Dict:
        """Step 2b: Analyze profile capabilities"""
        print("\n=== STEP 2b: Analyzing Profile Capabilities ===")
        
        if not self.user_profile:
            raise ValueError("User profile must be ingested first. Run step1_ingest_profile.")
        
        self.profile_capabilities = self.profile_analyzer.analyze_profile_capabilities(self.user_profile)
        
        print(f"✓ Profile capabilities extracted")
        print(f"  Core Skills: {len(self.profile_capabilities.get('core_skills', []))}")
        print(f"  Technologies: {len(self.profile_capabilities.get('technologies', []))}")
        print(f"  Projects: {len(self.profile_capabilities.get('projects', []))}")
        print(f"  Domain Expertise: {', '.join(self.profile_capabilities.get('domain_expertise', [])[:5])}")
        
        return self.profile_capabilities
    
    def step3_ingest_job_description(self, jd_input: str) -> Dict:
        """
        Step 3: Ingest and parse job description
        Accepts either a URL or raw text
        """
        print("\n=== STEP 3: Ingesting Job Description ===")
        
        if jd_input.startswith(('http://', 'https://')):
            print(f"Detected URL, extracting job description...")
        else:
            print(f"Processing job description text...")
        
        self.job_description = self.job_parser.parse_job_description(jd_input)
        
        jd_keywords = self.job_parser.get_keywords_for_embedding(self.job_description)
        self.alignment_engine.set_job_description(jd_keywords)
        
        print(f"✓ Job description parsed: {self.job_description.get('role', 'N/A')}")
        print(f"  Skills: {len(self.job_description.get('skills', []))}")
        print(f"  Requirements: {len(self.job_description.get('requirements', []))}")
        
        if self.profile_capabilities:
            print("\n=== STEP 3b: Matching Profile with Job Requirements ===")
            self.match_analysis = self.profile_analyzer.match_profile_with_job(
                profile_capabilities=self.profile_capabilities,
                job_description=self.job_description
            )
            print(f"✓ Profile matched: {self.match_analysis.get('match_score', 0)}% match score")
        
        return self.job_description
    
    def step4_analyze_resume(self, resume_path: str) -> list:
        """Step 4: Analyze current resume bullets"""
        print("\n=== STEP 4: Analyzing Resume ===")
        
        openai_api_key = self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        parsing_model = self.config.get("openai_settings", {}).get("parsing_model") or \
                       self.config.get("openai_settings", {}).get("model", "gpt-4o-mini")
        
        self.resume_parser = ResumeParser(
            tex_file_path=resume_path,
            api_key=openai_api_key,
            model=parsing_model
        )
        self.resume_parser.load_resume()
        bullets = self.resume_parser.extract_bullets()
        
        analyses = self.alignment_engine.analyze_all_bullets(
            bullets=bullets,
            user_profile_data=self.user_profile,
            match_analysis=self.match_analysis
        )
        
        decision_counts = {"KEEP": 0, "REWRITE": 0, "DE_EMPHASIZE": 0, "ADD": 0}
        for analysis in analyses:
            decision = analysis["decision"]
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        print(f"✓ Analyzed {len(bullets)} bullets")
        print(f"  Keep: {decision_counts['KEEP']}")
        print(f"  Rewrite: {decision_counts['REWRITE']}")
        print(f"  Add (new): {decision_counts.get('ADD', 0)}")
        print(f"  De-emphasize: {decision_counts['DE_EMPHASIZE']}")
        
        return analyses
    
    def step5_rewrite_bullets(self, analyses: list) -> list:
        """Step 5: Rewrite bullets via OpenAI API"""
        print("\n=== STEP 5: Rewriting Bullets ===")
        
        jd_keywords = self.job_parser.get_keywords_for_embedding(self.job_description)
        
        rewritten_count = 0
        for analysis in analyses:
            if analysis["decision"] == "REWRITE":
                bullet = analysis["bullet"]
                relevant_entries = analysis.get("relevant_entries", [])
                
                print(f"  Rewriting: {bullet['text'][:50]}...")
                
                rewritten_text = self.rewrite_engine.rewrite_bullet(
                    bullet=bullet,
                    job_keywords=jd_keywords,
                    relevant_profile_entries=relevant_entries,
                    match_analysis=self.match_analysis,
                    profile_capabilities=self.profile_capabilities
                )
                
                self.resume_parser.replace_bullet(bullet, rewritten_text)
                analysis["rewritten_text"] = rewritten_text
                rewritten_count += 1
            
            elif analysis["decision"] == "DE_EMPHASIZE":
                pass
        
        print(f"✓ Rewrote {rewritten_count} bullets")
        return analyses
    
    def step6_generate_documents(self, role_match_score: float) -> Dict:
        return {}
    
    def step7_commit_to_github(self, analyses: list, role_match_score: float, 
                               documents: Dict, commit_message: Optional[str] = None):
        """Step 7: Commit and push changes to GitHub"""
        print("\n=== STEP 7: Committing to GitHub ===")
        
        if not commit_message:
            role = self.job_description.get("role", "position")
            commit_message = f"Optimize resume for {role} (Match: {role_match_score}%)"
        
        output_dir = self.config.get("output", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        resume_path = self.config.get("repository", {}).get("resume_file", "main.tex")
        self.resume_parser.save_resume(os.path.join(output_dir, resume_path))
        
        output_data = {
            "role_match_score": role_match_score,
            "job_description": self.job_description,
            "profile_capabilities": self.profile_capabilities if self.profile_capabilities else {},
            "match_analysis": {
                "match_score": self.match_analysis.get("match_score", 0) if self.match_analysis else 0,
                "skill_matches": self.match_analysis.get("skill_matches", {}) if self.match_analysis else {},
                "missing_skills": self.match_analysis.get("missing_skills", []) if self.match_analysis else [],
                "strengths": self.match_analysis.get("strengths", []) if self.match_analysis else [],
                "recommendations": self.match_analysis.get("recommendations", []) if self.match_analysis else []
            } if self.match_analysis else {},
            "analyses": [
                {
                    "bullet_text": a["bullet"]["text"],
                    "decision": a["decision"],
                    "jd_similarity": a["jd_similarity"],
                    "profile_alignment": a.get("profile_alignment", 0),
                    "reasoning": a["reasoning"],
                    "rewritten_text": a.get("rewritten_text")
                }
                for a in analyses
            ],
            "documents": documents
        }
        
        json_path = os.path.join(output_dir, 
                                self.config.get("output", {}).get("json_output", "analysis_results.json"))
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved all outputs to {output_dir}/")
        
        if self.config.get("repository", {}).get("owner") and self.github_integration.token:
            try:
                self.github_integration.commit_and_push(
                    file_path=os.path.join(output_dir, resume_path),
                    commit_message=commit_message,
                    branch=self.config.get("repository", {}).get("branch", "main")
                )
                print(f"✓ Committed to GitHub: {commit_message}")
            except Exception as e:
                print(f"  Warning: GitHub commit failed: {e}")
        
        return output_data
    
    def run_full_workflow(self, jd_input: str, resume_path: str, 
                         scholar_id: Optional[str] = None,
                         author_name: Optional[str] = None,
                         rebuild_embeddings: bool = False) -> Dict:
        """
        Run the complete workflow from start to finish
        jd_input can be either a URL or raw text
        """
        print("\n" + "="*60)
        print("ATS RESUME OPTIMIZER - FULL WORKFLOW")
        print("="*60)
        
        self.step1_ingest_profile(scholar_id=scholar_id, author_name=author_name)
        self.step2_create_embedding_store(rebuild=rebuild_embeddings)
        self.step2b_analyze_profile()
        self.step3_ingest_job_description(jd_input)
        analyses = self.step4_analyze_resume(resume_path)
        
        if self.match_analysis and self.match_analysis.get("match_score"):
            role_match_score = self.match_analysis["match_score"]
        else:
            role_match_score = self.alignment_engine.calculate_role_match_score(analyses)
        
        print(f"\nRole Match Score: {role_match_score}%")
        
        if self.match_analysis:
            print(f"  Profile Strengths: {', '.join(self.match_analysis.get('strengths', [])[:5])}")
            if self.match_analysis.get('missing_skills'):
                print(f"  Missing Skills: {', '.join(self.match_analysis['missing_skills'][:5])}")
        
        analyses = self.step5_rewrite_bullets(analyses)
        documents = {}
        output_data = self.step7_commit_to_github(analyses, role_match_score, documents)
        
        print("\n" + "="*60)
        print("✓ WORKFLOW COMPLETE")
        print("="*60)
        
        return output_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <job_description_url_or_file> <resume_file> [scholar_id/author_name]")
        print("\nExamples:")
        print("  python main.py https://linkedin.com/jobs/view/123456 resume.tex")
        print("  python main.py job_description.txt resume.tex")
        sys.exit(1)
    
    jd_input = sys.argv[1]
    resume_file = sys.argv[2]
    scholar_arg = sys.argv[3] if len(sys.argv) > 3 else None
    
    if jd_input.startswith(('http://', 'https://')):
        jd_text = jd_input
        print(f"Job description URL: {jd_input}")
    else:
        try:
            with open(jd_input, 'r') as f:
                jd_text = f.read()
            print(f"Job description file: {jd_input}")
        except FileNotFoundError:
            print(f"Error: File not found: {jd_input}")
            sys.exit(1)
    
    optimizer = ATSResumeOptimizer()
    results = optimizer.run_full_workflow(
        jd_input=jd_text,
        resume_path=resume_file,
        author_name=scholar_arg if scholar_arg else None
    )
    
    print(f"\nResults saved. Role match score: {results['role_match_score']}%")

