"""
GitHub Integration Module
Handles committing and pushing changes to GitHub repository
"""

import os
from pathlib import Path
from github import Github
from typing import Optional


class GitHubIntegration:
    """Handles GitHub repository operations"""
    
    def __init__(self, token: str, repo_owner: str, repo_name: str, branch: str = "main"):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.github = Github(token) if token else None
        self.repo = None
        
        if self.github:
            try:
                self.repo = self.github.get_user(self.repo_owner).get_repo(self.repo_name)
            except Exception as e:
                print(f"Warning: Could not access GitHub repo: {e}")
    
    def commit_and_push(self, file_path: str, commit_message: str, branch: str = None, repo_path: str = None):
        """Commit and push a file to GitHub repository"""
        if not self.token:
            raise ValueError("GitHub token not provided")
        
        branch = branch or self.branch
        
        # Read file content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine path in repository
        if repo_path is None:
            file_name = os.path.basename(file_path)
            repo_path = file_name  # Assuming file goes to root, adjust if needed
        
        try:
            # Try to get existing file
            try:
                contents = self.repo.get_contents(repo_path, ref=branch)
                # Update existing file
                self.repo.update_file(
                    path=repo_path,
                    message=commit_message,
                    content=content,
                    sha=contents.sha,
                    branch=branch
                )
                print(f"Updated {repo_path} in GitHub")
            except:
                # Create new file
                self.repo.create_file(
                    path=repo_path,
                    message=commit_message,
                    content=content,
                    branch=branch
                )
                print(f"Created {repo_path} in GitHub")
                
        except Exception as e:
            raise Exception(f"Failed to commit to GitHub: {e}")
    
    def commit_multiple_files(self, files: dict, commit_message: str, branch: str = None):
        """Commit multiple files in a single commit"""
        if not self.token:
            raise ValueError("GitHub token not provided")
        
        branch = branch or self.branch
        
        # Get base commit SHA
        try:
            base_sha = self.repo.get_branch(branch).commit.sha
        except:
            raise Exception(f"Branch {branch} not found")
        
        # Prepare file changes
        file_changes = []
        
        for repo_path, local_path in files.items():
            if not os.path.exists(local_path):
                print(f"Warning: File not found: {local_path}")
                continue
            
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to get existing file SHA
            try:
                existing_file = self.repo.get_contents(repo_path, ref=branch)
                file_changes.append({
                    "path": repo_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": existing_file.sha,
                    "content": content
                })
            except:
                # New file
                file_changes.append({
                    "path": repo_path,
                    "mode": "100644",
                    "type": "blob",
                    "content": content
                })
        
        if not file_changes:
            print("No files to commit")
            return
        
        try:
            # Create tree
            tree = self.repo.create_git_tree(
                tree=file_changes,
                base_tree=self.repo.get_git_tree(sha=base_sha)
            )
            
            # Create commit
            commit = self.repo.create_git_commit(
                message=commit_message,
                tree=tree,
                parents=[self.repo.get_git_commit(sha=base_sha)]
            )
            
            # Update branch reference
            ref = self.repo.get_git_ref(f"heads/{branch}")
            ref.edit(sha=commit.sha)
            
            print(f"Committed {len(file_changes)} files to GitHub")
            
        except Exception as e:
            raise Exception(f"Failed to commit multiple files: {e}")
    
    def get_file_content(self, file_path: str, branch: str = None) -> str:
        """Get file content from GitHub repository"""
        if not self.repo:
            raise ValueError("Repository not initialized")
        
        branch = branch or self.branch
        
        try:
            contents = self.repo.get_contents(file_path, ref=branch)
            return contents.decoded_content.decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to get file from GitHub: {e}")

