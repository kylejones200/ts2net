#!/usr/bin/env python3
"""Create a GitHub release from CHANGELOG.md"""

import re
import subprocess
import sys
from pathlib import Path

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
    # Remove markdown code blocks if present and clean up
    notes = re.sub(r'```.*?```', '', notes, flags=re.DOTALL)
    return notes

def create_release(version, notes):
    """Create GitHub release using gh CLI"""
    cmd = ['gh', 'release', 'create', version, '--title', version, '--notes', notes]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating release: {result.stderr}", file=sys.stderr)
        return False
    
    print(f"✅ Successfully created release {version}")
    print(result.stdout)
    return True

if __name__ == '__main__':
    version = 'v0.5.0'
    changelog_path = Path(__file__).parent.parent / 'CHANGELOG.md'
    
    try:
        notes = extract_release_notes(changelog_path, '0.5.0')
        print(f"Creating release {version}...")
        print(f"Release notes:\n{notes[:200]}...\n")
        create_release(version, notes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


