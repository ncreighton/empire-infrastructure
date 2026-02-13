# Claude Code Integration Guide

## Project Location
```
D:\Claude Code Projects\
```

## When to Use This Tool

Trigger this tool when user requests:
- "Create an image for [site]"
- "Generate featured image for [article]"
- "Make Pinterest pin for [topic]"
- "Upload image to [site]"
- "Set featured image on post [ID]"
- Any article image generation task

---

## Quick Commands

### Generate + Upload (Most Common)
```bash
cd "D:\Claude Code Projects"
python article_images_pipeline.py --site SITE_ID --title "TITLE" --enhanced
```

### Generate + Upload + Set Featured
```bash
python article_images_pipeline.py --site SITE_ID --title "TITLE" --post-id POST_ID --enhanced
```

### Single Image Type
```bash
python article_images_pipeline.py --site SITE_ID --title "TITLE" --type blog_featured --enhanced
```

### Local Only (No Upload)
```bash
python article_images_pipeline.py --site SITE_ID --title "TITLE" --enhanced --no-upload
```

---

## Site IDs (Use Exactly)

```
witchcraftforbeginners
smarthomewizards
mythicalarchives
bulletjournals
wealthfromai
aidiscoverydigest
aiinactionhub
pulsegearreviews
wearablegearreviews
smarthomegearreviews
clearainews
theconnectedhaven
manifestandalign
familyflourish
```

---

## Image Types

```
blog_featured    (1200×630)  - Default, OG images
pinterest_pin    (1000×1500) - Pinterest vertical
instagram_post   (1080×1080) - Instagram square
facebook_post    (1200×630)  - Facebook
twitter_post     (1600×900)  - Twitter/X
```

---

## Arguments Reference

| Argument | Required | Description |
|----------|----------|-------------|
| `--site` | Yes | Site ID from list above |
| `--title` | Yes | Headline text (auto-wraps) |
| `--subtitle` | No | Secondary text below headline |
| `--type` | No | Single image type (default: all 5) |
| `--post-id` | No | WordPress post ID for featured image |
| `--enhanced` | No | Use visual patterns (recommended) |
| `--no-upload` | No | Generate locally only |

---

## Programmatic Integration

### Import and call directly
```python
import sys
sys.path.insert(0, r'D:\Claude Code Projects')

from enhanced_image_gen import create_enhanced_image

# Generate single image
create_enhanced_image(
    site_id="witchcraftforbeginners",
    title="Full Moon Ritual",
    image_type="blog_featured",
    subtitle="Begin your magical journey",
    output_path="/tmp/moon-ritual.png"
)
```

### Generate batch
```python
from simple_image_gen import create_batch

files = create_batch(
    site_id="smarthomewizards",
    title="Smart Home Guide",
    output_dir="/tmp/images/"
)
# Returns: {'blog_featured': '/tmp/images/blog_featured-smart-home-guide.png', ...}
```

### Load site config
```python
import json

def get_site_config(site_id):
    with open(r'D:\Claude Code Projects\config\sites.json') as f:
        data = json.load(f)
        sites = data.get('sites', data)
        return sites.get(site_id)

config = get_site_config('witchcraftforbeginners')
# Returns: {domain, wp_user, wp_app_password, brand_name, colors, etc.}
```

---

## Extending This Tool

### Add a new site
Edit `config/sites.json`:
```json
{
  "sites": {
    "newsite": {
      "domain": "newsite.com",
      "wp_user": "Username",
      "wp_app_password": "xxxx xxxx xxxx xxxx xxxx xxxx",
      "brand_name": "NewSite",
      "tagline": "Site tagline",
      "primary_color": "#FF0000",
      "secondary_color": "#00FF00",
      "gradient_start": "#000000",
      "gradient_end": "#333333",
      "text_color": "#FFFFFF",
      "pattern": "stars"
    }
  }
}
```

### Add a new pattern
Edit `enhanced_image_gen.py`, add to `draw_pattern()` function:
```python
elif pattern == "newpattern":
    # Your pattern drawing code
    pass
```

### Add a new image type
Edit both generators, add to `IMAGE_TYPES` dict:
```python
IMAGE_TYPES = {
    # existing types...
    "youtube_thumbnail": (1280, 720),
}
```

---

## Workflow Integration Examples

### Content Pipeline Integration
```python
def on_article_published(site_id, title, post_id):
    """Called when new article is published"""
    import subprocess
    
    subprocess.run([
        "python", r"D:\Claude Code Projects\article_images_pipeline.py",
        "--site", site_id,
        "--title", title,
        "--post-id", str(post_id),
        "--enhanced"
    ])
```

### Batch Processing
```python
articles = [
    {"site": "witchcraftforbeginners", "title": "Moon Phases", "post_id": 123},
    {"site": "smarthomewizards", "title": "Alexa Tips", "post_id": 456},
]

for article in articles:
    create_enhanced_image(
        site_id=article["site"],
        title=article["title"],
        image_type="blog_featured",
        output_path=f"/tmp/{article['post_id']}.png"
    )
```

---

## Troubleshooting

### "Site not found"
Check site ID matches exactly (lowercase, no spaces).

### "HTTP 401"
WordPress app password expired. Regenerate in wp-admin → Users → Profile → Application Passwords.

### "Font not found"
System uses DejaVu Sans. Install: `apt-get install fonts-dejavu`

### Images look wrong
Use `--enhanced` flag for proper branded patterns.

---

## Output Locations

- **Generated images:** `/tmp/article-images-{timestamp}/`
- **Uploaded to WordPress:** `https://{domain}/wp-content/uploads/{year}/{month}/{filename}.png`

---

## Dependencies Check
```bash
python -c "from PIL import Image; import requests; print('✅ Dependencies OK')"
```

If missing:
```bash
pip install Pillow requests
```

---

## API Cost Optimization Rules

### Model Selection (MANDATORY)
When generating code that calls Anthropic's API:

1. **Default to Sonnet** (`claude-sonnet-4-20250514`) for most tasks
2. **Use Haiku** (`claude-haiku-4-5-20251001`) for:
   - Classification tasks
   - Intent detection
   - Simple data extraction
   - Yes/no decisions
   - Formatting/conversion
   - Tag generation
3. **Reserve Opus** (`claude-opus-4-20250514`) ONLY for:
   - Complex multi-step reasoning
   - Critical business decisions
   - Nuanced editorial judgment

### Prompt Caching (ALWAYS ENABLE)
When system prompts exceed 2,048 tokens, ALWAYS use cache_control:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": user_input}]
)
```

### Token Limits
| Output Type | max_tokens |
|-------------|------------|
| Yes/no, classification | 50-100 |
| Short response | 200-500 |
| Article section | 1000-2000 |
| Full article | 3000-4096 |

### Quick Reference
```
Model Strings (Dec 2025):
- claude-haiku-4-5-20251001    → Simple tasks
- claude-sonnet-4-20250514     → Default
- claude-opus-4-20250514       → Complex only

Pricing per 1M tokens:
- Haiku:  $0.80 in / $4.00 out
- Sonnet: $3.00 in / $15.00 out
- Opus:   $15.00 in / $75.00 out
- Cache reads: 90% discount
- Batch API: 50% discount
```
