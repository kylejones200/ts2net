#!/usr/bin/env python3
"""Create a GitHub release using GitHub API"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

def get_repo_info():
    """Get repository owner and name from git remote"""
    result = subprocess.run(
        ['git', 'config', '--get', 'remote.origin.url'],
        capture_output=True, text=True
    )
    url = result.stdout.strip()
    # Handle both https and ssh URLs
    if 'github.com' in url:
        if url.startswith('https://'):
            match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', url)
        else:
            match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', url)
        if match:
            return match.group(1), match.group(2)
    raise ValueError("Could not determine repository from git remote")

def extract_release_notes(changelog_path, version):
    """Extract release notes for a specific version from CHANGELOG.md"""
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Find the section for this version
    pattern = rf'## \[{re.escape(version)}\][^\n]*\n(.*?)(?=\n## \[|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not find release notes for version {version}")
    
    notes = match.group(1).strip()
    return notes

def create_release_api(owner, repo, version, notes, token):
    """Create GitHub release using REST API"""
    import urllib.request
    import urllib.error
    
    url = f'https://api.github.com/repos/{owner}/{repo}/releases'
    data = {
        'tag_name': version,
        'name': version,
        'body': notes,
        'draft': False,
        'prerelease': False
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode())
    req.add_header('Authorization', f'token {token}')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    req.add_header('Content-Type', 'application/json')
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"✅ Successfully created release {version}")
            print(f"Release URL: {result.get('html_url', 'N/A')}")
            return True
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Error creating release: {e.code} {e.reason}", file=sys.stderr)
        print(f"Response: {error_body}", file=sys.stderr)
        return False

if __name__ == '__main__':
    version = 'v0.5.0'
    changelog_path = Path(__file__).parent.parent / 'CHANGELOG.md'
    
    # Try to get token from environment
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    
    if not token:
        print("Error: GITHUB_TOKEN or GH_TOKEN environment variable not set", file=sys.stderr)
        print("Set it with: export GITHUB_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)
    
    try:
        owner, repo = get_repo_info()
        notes = extract_release_notes(changelog_path, '0.5.0')
        print(f"Creating release {version} for {owner}/{repo}...")
        create_release_api(owner, repo, version, notes, token)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


