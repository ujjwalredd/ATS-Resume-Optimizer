# ATS Resume Optimizer

AI-powered tool that automatically optimizes LaTeX resumes based on job descriptions. Analyzes your profile from GitHub, LinkedIn, and Google Scholar, then intelligently rewrites resume bullets to match job requirements while maintaining factual accuracy.

![Dashboard](dashboard.png)

## Features

- URL Extraction: Extract job descriptions directly from LinkedIn, Indeed, Glassdoor URLs
- Profile-First Analysis: Analyzes your capabilities before optimizing
- Smart Matching: Matches profile with job requirements, identifies strengths and gaps
- Evidence-Based: Only uses your actual profile data - never invents experiences
- AI-Powered: Uses OpenAI for parsing and rewriting
- Role Matching: Provides 0-100% match score
- GitHub Integration: Auto-commits optimized resume to your repo
- Web Dashboard: Streamlit interface for easy use

## Quick Start

### Installation

```bash
git clone https://github.com/ujjwalredd/ATS-Resume-Optimizer.git
cd Resume_AI
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys
```

### Usage

#### Option 1: Web Dashboard (Recommended)

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 and follow the guided workflow:
1. Profile Setup: Enter GitHub, Scholar info and resume repo URL
2. Job Analysis: Paste job posting URL or text
3. View Results: See match score and analysis
4. Download: Get optimized resume and analysis files

#### Option 2: Command Line

```bash
# Using job posting URL
python main.py https://linkedin.com/jobs/view/123456 resume.tex

# Using text file
python main.py job_description.txt resume.tex
```

#### Option 3: Python API

```python
from main import ATSResumeOptimizer

optimizer = ATSResumeOptimizer("config.yaml")

results = optimizer.run_full_workflow(
    jd_input="https://linkedin.com/jobs/view/123456",
    resume_path="resume.tex",
    author_name="Your Name"
)

print(f"Match Score: {results['role_match_score']}%")
```

## Configuration

Create config.yaml from config.yaml.example:

```yaml
openai:
  api_key: "your-openai-api-key"

github:
  token: "your-github-token"
  username: "your-github-username"

repository:
  owner: "your-github-username"
  name: "resume-repo"
  resume_file: "main.tex"
  branch: "main"

openai_settings:
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 500
```

Or use environment variables:
```bash
export OPENAI_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
export GITHUB_USERNAME="your-username"
```

## Workflow

1. Profile Ingestion: Fetches from GitHub, LinkedIn, Google Scholar
2. Profile Analysis: Extracts capabilities, skills, experiences
3. Job Parsing: Extracts role, skills, requirements from URL/text
4. Profile-Job Matching: Identifies strengths, gaps, recommendations
5. Resume Analysis: Compares bullets with job requirements
6. Decision Making: KEEP/REWRITE/ADD/DE_EMPHASIZE based on profile
7. Rewriting: Optimizes bullets using AI with profile evidence
8. GitHub Push: Commits optimized resume to your repo

## Output Structure

Each optimization creates a timestamped folder:

```
output/job_YYYYMMDD_HHMMSS/
├── analysis_results.json  # Complete analysis with match score, bullet decisions, recommendations
└── main.tex              # Optimized resume
```

Files are automatically pushed to your GitHub repository.

## Architecture

Modular Python codebase:

- profile_ingester.py: Fetches GitHub, LinkedIn, Scholar data
- profile_analyzer.py: Analyzes profile capabilities
- embedding_store.py: Vector embeddings with sentence-transformers + FAISS
- job_parser.py: AI-powered job description parsing
- resume_parser.py: AI-powered LaTeX resume parsing
- alignment_engine.py: Analyzes bullets vs job requirements
- rewrite_engine.py: Rewrites bullets using OpenAI
- github_integration.py: GitHub commits and pushes
- main.py: Workflow orchestrator
- dashboard.py: Streamlit web interface

## Key Features Explained

### Profile-First Analysis
The system analyzes your profile BEFORE optimizing. It understands what you have (skills, projects, experiences) and only suggests additions supported by actual evidence.

### URL Extraction
Automatically extracts job descriptions from:
- LinkedIn Jobs
- Indeed
- Glassdoor
- Any other job board (generic extraction)

### Smart Matching
Compares your profile with job requirements to identify:
- Strengths: What you have that matches
- Gaps: What's missing (for awareness)
- Recommendations: Specific actions with evidence

### Evidence-Based Rewriting
Every rewritten bullet:
- Uses keywords from job description
- Incorporates your actual profile data
- Includes metrics and achievements from your projects
- Never invents skills or experiences

## Requirements

- Python 3.8+
- OpenAI API key
- GitHub personal access token
- Internet connection
- LaTeX resume file (.tex format)

## Troubleshooting

Dashboard won't start?
```bash
pip install streamlit
streamlit run dashboard.py
```

GitHub fetch fails?
- Verify repo URL is correct
- Check token has repo scope
- Ensure branch name exists

Job extraction fails?
- Try copying job description text instead
- Some sites require authentication (LinkedIn personal jobs)

Import errors?
```bash
pip install -r requirements.txt
```

Embedding store issues?
```python
optimizer.step2_create_embedding_store(rebuild=True)
```

## Safety & Ethics

- No Invention: Never adds skills/experiences you don't have
- Factual Only: Uses only your actual profile data
- Human Review: Always review AI output before submitting
- Privacy: API keys stored securely, never committed

## License

MIT License

## Disclaimer

This tool provides assistance only. Always review and verify all AI-generated content before submitting job applications. The authors are not responsible for any misuse or incorrect information.
