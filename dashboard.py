"""
Streamlit Dashboard for ATS Resume Optimizer
Interactive web interface for resume optimization
"""

import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ATSResumeOptimizer
from profile_ingester import ProfileIngester


# Page configuration
st.set_page_config(
    page_title="ATS Resume Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'profile_data' not in st.session_state:
        st.session_state.profile_data = None
    if 'profile_capabilities' not in st.session_state:
        st.session_state.profile_capabilities = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'github_repo' not in st.session_state:
        st.session_state.github_repo = None
    if 'resume_path' not in st.session_state:
        st.session_state.resume_path = None
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = None


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📄 ATS Resume Optimizer</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                  value=os.getenv("OPENAI_API_KEY", ""),
                                  help="Required for AI parsing and rewriting")
        github_token = st.text_input("GitHub Token", type="password",
                                    value=os.getenv("GITHUB_TOKEN", ""),
                                    help="Required for profile ingestion and auto-push")
        
        # Save API keys to environment
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        rebuild_embeddings = st.checkbox("Rebuild Embeddings", value=False,
                                        help="Force rebuild of embedding store")
        
        # Config file path
        config_path = st.text_input("Config File Path", value="config.yaml",
                                   help="Path to config.yaml file (optional)")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Profile Setup", "💼 Job Analysis", "📊 Results", "📝 Resume Output"])
    
    # Tab 1: Profile Setup
    with tab1:
        st.markdown('<div class="section-header">Step 1: Profile Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("GitHub Profile")
            github_username = st.text_input("GitHub Username", 
                                           help="Your GitHub username for profile ingestion")
            github_profile_url = st.text_input("GitHub Profile URL (optional)",
                                              help="Alternative: Provide full GitHub profile URL")
        
        with col2:
            st.subheader("Google Scholar")
            scholar_author_name = st.text_input("Author Name",
                                               help="Your name as it appears on Google Scholar")
            scholar_id = st.text_input("Scholar ID (optional)",
                                      help="Your Google Scholar profile ID if known")
        
        st.markdown("---")
        st.subheader("LinkedIn Profile")
        linkedin_url = st.text_input("LinkedIn Profile URL (optional)",
                                    help="Note: LinkedIn extraction requires API access or manual input")
        
        st.markdown("---")
        st.subheader("Resume Repository")
        resume_repo_url = st.text_input("GitHub Resume Repo URL",
                                       help="e.g., https://github.com/username/resume-repo")
        resume_file_name = st.text_input("Resume File Name", value="main.tex",
                                        help="Name of your LaTeX resume file in the repo")
        resume_branch = st.text_input("Branch Name", value="main",
                                     help="GitHub branch to use")
        
        # Store in session state
        if resume_repo_url:
            # Parse GitHub repo URL
            if "github.com" in resume_repo_url:
                parts = resume_repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
                if len(parts) >= 2:
                    repo_owner = parts[0]
                    repo_name = parts[1].split(".git")[0].split("/")[0]
                    st.session_state.github_repo = {
                        "owner": repo_owner,
                        "name": repo_name,
                        "branch": resume_branch,
                        "file": resume_file_name
                    }
        
        # Button to ingest profile
        if st.button("🔍 Analyze Profile", type="primary", use_container_width=True):
            if not openai_key:
                st.error("❌ Please provide OpenAI API Key in the sidebar")
            elif not github_token:
                st.error("❌ Please provide GitHub Token in the sidebar")
            elif not github_username and not github_profile_url:
                st.error("❌ Please provide GitHub username or profile URL")
            else:
                with st.spinner("Analyzing your profile... This may take a few minutes."):
                    try:
                        # Initialize optimizer
                        config = {
                            "github": {"token": github_token, "username": github_username or "unknown"},
                            "openai": {"api_key": openai_key},
                            "openai_settings": {"model": "gpt-4o-mini"},
                            "repository": st.session_state.github_repo or {}
                        }
                        
                        # Save config temporarily
                        import yaml
                        temp_config_path = "temp_config.yaml"
                        with open(temp_config_path, 'w') as f:
                            yaml.dump(config, f)
                        
                        st.session_state.optimizer = ATSResumeOptimizer(temp_config_path)
                        
                        # Step 1: Ingest profile
                        st.info("📥 Ingesting profile data from GitHub and Scholar...")
                        profile_data = st.session_state.optimizer.step1_ingest_profile(
                            scholar_id=scholar_id if scholar_id else None,
                            author_name=scholar_author_name if scholar_author_name else None
                        )
                        st.session_state.profile_data = profile_data
                        
                        # Step 2: Create embedding store
                        st.info("🔢 Creating embedding store...")
                        st.session_state.optimizer.step2_create_embedding_store(rebuild=rebuild_embeddings)
                        
                        # Step 2b: Analyze profile
                        st.info("🧠 Analyzing profile capabilities...")
                        profile_capabilities = st.session_state.optimizer.step2b_analyze_profile()
                        st.session_state.profile_capabilities = profile_capabilities
                        
                        st.success("✅ Profile analysis complete!")
                        
                        # Display profile summary
                        st.markdown("### Profile Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Skills", len(profile_capabilities.get('core_skills', [])))
                        with col2:
                            st.metric("Technologies", len(profile_capabilities.get('technologies', [])))
                        with col3:
                            st.metric("Projects", len(profile_capabilities.get('projects', [])))
                        
                        # Display capabilities
                        if profile_capabilities.get('core_skills'):
                            st.markdown("#### Core Skills")
                            st.write(", ".join(profile_capabilities['core_skills'][:20]))
                        
                        if profile_capabilities.get('domain_expertise'):
                            st.markdown("#### Domain Expertise")
                            st.write(", ".join(profile_capabilities['domain_expertise'][:10]))
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing profile: {str(e)}")
                        st.exception(e)
    
    # Tab 2: Job Analysis
    with tab2:
        if st.session_state.optimizer is None:
            st.warning("⚠️ Please complete Profile Setup first (Tab 1)")
        else:
            st.markdown('<div class="section-header">Step 2: Job Description</div>', unsafe_allow_html=True)
            
            st.subheader("Job Posting")
            job_input = st.text_input("Job Posting URL or Paste Text Here",
                                     help="Paste the job posting URL (LinkedIn, Indeed, etc.) or the full job description text")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                analyze_job_btn = st.button("🚀 Optimize Resume for This Job", type="primary", use_container_width=True)
            
            if analyze_job_btn:
                if not job_input:
                    st.error("❌ Please provide a job posting URL or text")
                elif not st.session_state.github_repo:
                    st.error("❌ Please provide GitHub Resume Repo URL in Profile Setup")
                else:
                    with st.spinner("Optimizing your resume... This may take 2-5 minutes."):
                        try:
                            # Step 3: Parse job description
                            st.info("📋 Parsing job description...")
                            job_desc = st.session_state.optimizer.step3_ingest_job_description(job_input)
                            st.session_state.job_description = job_desc
                            
                            # Display job info
                            st.markdown("### Job Information")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Role", job_desc.get('role', 'N/A'))
                            with col2:
                                st.metric("Company", job_desc.get('company', 'N/A'))
                            with col3:
                                st.metric("Location", job_desc.get('location', 'N/A'))
                            
                            # Download resume from GitHub if needed
                            resume_file_name = st.session_state.github_repo.get("file", "main.tex")
                            resume_path = f"temp_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
                            st.info("📥 Fetching resume from GitHub...")
                            
                            # Try to get resume from GitHub or use local file
                            try:
                                from github_integration import GitHubIntegration
                                gh_integration = GitHubIntegration(
                                    token=github_token,
                                    repo_owner=st.session_state.github_repo["owner"],
                                    repo_name=st.session_state.github_repo["name"],
                                    branch=st.session_state.github_repo["branch"]
                                )
                                resume_content = gh_integration.get_file_content(
                                    resume_file_name
                                )
                                with open(resume_path, 'w') as f:
                                    f.write(resume_content)
                                st.session_state.resume_path = resume_path
                                st.success(f"✅ Fetched {resume_file_name} from GitHub")
                            except Exception as e:
                                st.warning(f"Could not fetch from GitHub: {e}. Using example resume.")
                                # Use example resume as fallback
                                if os.path.exists("example_main.tex"):
                                    resume_path = "example_main.tex"
                                    resume_file_name = "example_main.tex"
                                    st.session_state.resume_path = resume_path
                                else:
                                    raise Exception("No resume file available. Please ensure your GitHub repo URL and file name are correct.")
                            
                            # Step 4: Analyze resume
                            st.info("🔍 Analyzing resume...")
                            analyses = st.session_state.optimizer.step4_analyze_resume(resume_path)
                            
                            # Calculate match score
                            if st.session_state.optimizer.match_analysis:
                                match_score = st.session_state.optimizer.match_analysis.get("match_score", 0)
                            else:
                                match_score = st.session_state.optimizer.alignment_engine.calculate_role_match_score(analyses)
                            
                            # Step 5: Rewrite bullets
                            st.info("✍️ Rewriting resume bullets...")
                            analyses = st.session_state.optimizer.step5_rewrite_bullets(analyses)
                            
                            # Step 6: Skip document generation (removed feature)
                            documents = {}
                            
                            # Prepare results
                            results = {
                                "role_match_score": match_score,
                                "job_description": job_desc,
                                "profile_capabilities": st.session_state.profile_capabilities,
                                "match_analysis": st.session_state.optimizer.match_analysis or {},
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
                                "documents": documents,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            st.session_state.results = results
                            
                            # Create output folder with timestamp
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_folder = f"output/job_{timestamp}"
                            os.makedirs(output_folder, exist_ok=True)
                            
                            # Save analysis results (changes, score, etc.)
                            analysis_file = os.path.join(output_folder, "analysis_results.json")
                            with open(analysis_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            
                            # Save modified resume
                            modified_resume_path = os.path.join(output_folder, resume_file_name)
                            st.session_state.optimizer.resume_parser.save_resume(modified_resume_path)
                            
                            # Store output folder in session state
                            st.session_state.output_folder = output_folder
                            
                            # Push modified resume to GitHub
                            st.info("🚀 Pushing to GitHub...")
                            try:
                                commit_message = f"Optimize resume for {job_desc.get('role', 'position')} (Match: {match_score}%)"
                                
                                # Read the modified resume content
                                with open(modified_resume_path, 'r') as f:
                                    resume_content = f.read()
                                
                                # Use GitHub integration to commit
                                from github_integration import GitHubIntegration
                                gh_integration = GitHubIntegration(
                                    token=github_token,
                                    repo_owner=st.session_state.github_repo["owner"],
                                    repo_name=st.session_state.github_repo["name"],
                                    branch=st.session_state.github_repo["branch"]
                                )
                                
                                # Commit the resume file to the same path in repo
                                gh_integration.commit_and_push(
                                    file_path=modified_resume_path,
                                    commit_message=commit_message,
                                    branch=st.session_state.github_repo["branch"],
                                    repo_path=st.session_state.github_repo["file"]  # Use original file path in repo
                                )
                                
                                st.success(f"✅ Pushed to GitHub: {commit_message}")
                                st.info(f"📁 Files saved in: {output_folder}")
                                st.info(f"  - analysis_results.json (changes, score, etc.)")
                                st.info(f"  - {resume_file_name} (modified resume)")
                                
                            except Exception as e:
                                st.warning(f"⚠️ GitHub push failed: {e}")
                                st.info(f"Files saved locally in: {output_folder}")
                                st.exception(e)
                            
                            st.success("✅ Resume optimization complete!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error optimizing resume: {str(e)}")
                            st.exception(e)
    
    # Tab 3: Results
    with tab3:
        if st.session_state.results is None:
            st.info("👈 Complete Job Analysis (Tab 2) to see results")
        else:
            results = st.session_state.results
            
            st.markdown('<div class="section-header">Optimization Results</div>', unsafe_allow_html=True)
            
            # Match Score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Role Match Score", f"{results['role_match_score']:.1f}%")
            with col2:
                rewritten = sum(1 for a in results['analyses'] if a.get('rewritten_text'))
                st.metric("Bullets Rewritten", rewritten)
            with col3:
                total = len(results['analyses'])
                st.metric("Total Bullets", total)
            
            # Match Analysis
            if results.get('match_analysis'):
                st.markdown("### Profile-Job Match Analysis")
                
                match_analysis = results['match_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if match_analysis.get('strengths'):
                        st.markdown("#### ✅ Strengths")
                        for strength in match_analysis['strengths'][:10]:
                            st.write(f"- {strength}")
                
                with col2:
                    if match_analysis.get('missing_skills'):
                        st.markdown("#### ⚠️ Missing Skills")
                        for skill in match_analysis['missing_skills'][:10]:
                            st.write(f"- {skill}")
            
            # Bullet Analysis
            st.markdown("### Bullet Point Analysis")
            
            decisions = {}
            for analysis in results['analyses']:
                decision = analysis['decision']
                decisions[decision] = decisions.get(decision, 0) + 1
            
            # Decision breakdown
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Keep", decisions.get('KEEP', 0))
            with col2:
                st.metric("Rewrite", decisions.get('REWRITE', 0))
            with col3:
                st.metric("De-emphasize", decisions.get('DE_EMPHASIZE', 0))
            with col4:
                st.metric("Add", decisions.get('ADD', 0))
            
            # Show rewritten bullets
            rewritten_bullets = [a for a in results['analyses'] if a.get('rewritten_text')]
            if rewritten_bullets:
                st.markdown("#### Rewritten Bullets")
                for i, bullet in enumerate(rewritten_bullets[:5], 1):
                    with st.expander(f"Bullet {i}: {bullet['bullet_text'][:60]}..."):
                        st.write("**Original:**")
                        st.write(bullet['bullet_text'])
                        st.write("**Rewritten:**")
                        st.write(bullet['rewritten_text'])
                        st.write("**Reasoning:**")
                        st.write(bullet['reasoning'])
            
            # Download results
            st.markdown("### Download Results")
            if hasattr(st.session_state, 'output_folder') and st.session_state.output_folder:
                output_folder = st.session_state.output_folder
                analysis_file = os.path.join(output_folder, "analysis_results.json")
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as f:
                        analysis_json = f.read()
                    st.download_button("📥 Download Analysis JSON", analysis_json, 
                                     "analysis_results.json", "application/json")
    
    # Tab 4: Resume Output
    with tab4:
        if st.session_state.results is None:
            st.info("👈 Complete Job Analysis (Tab 2) to see resume output")
        else:
            if st.session_state.optimizer and st.session_state.optimizer.resume_parser:
                st.markdown('<div class="section-header">Modified Resume</div>', unsafe_allow_html=True)
                
                # Show resume content
                resume_content = st.session_state.optimizer.resume_parser.content
                
                st.text_area("LaTeX Resume Content", resume_content, height=600)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Resume (.tex)", resume_content,
                                     "optimized_resume.tex", "text/plain")
                
                # Find saved resume file
                if hasattr(st.session_state, 'output_folder') and st.session_state.output_folder:
                    output_folder = st.session_state.output_folder
                    resume_file_path = os.path.join(output_folder, resume_file_name)
                    if os.path.exists(resume_file_path):
                        with open(resume_file_path, 'r') as f:
                            resume_file_content = f.read()
                        with col2:
                            st.download_button(f"📥 Download {resume_file_name}", resume_file_content,
                                             resume_file_name, "text/plain")
                    
                    # Show output folder info
                    st.markdown("### Output Folder")
                    st.info(f"📁 All files saved in: `{output_folder}`")
                    st.code(f"""
{output_folder}/
├── analysis_results.json  # Changes, scores, analysis
└── {resume_file_name}     # Modified resume
                    """)


if __name__ == "__main__":
    main()

