import sys
import re

def clean_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Strip <div class="math-box"> wrappers which break GitHub Markdown math parsing
    content = re.sub(r'<div class="math-box">\s*(.*?)\s*</div>', r'\1', content, flags=re.DOTALL)

    with open(path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    clean_file('presentation_slides.md')
    clean_file('README.md')
