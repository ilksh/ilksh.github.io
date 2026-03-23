# README — Authoring Workflow (Markdown + Static Pages)

This repository is organized around a convention-based pipeline:
you write content in Markdown with consistent front matter,
and then the appropriate `_template.html` wrapper turns it into a static HTML page under the target section.

Most “correctness” here is therefore about following the expected naming and routing conventions.

---

## Step 1: Author the Markdown Source
- Create `posts/[slug].md`.

```markdown
---
title: Article title
category: Category name
date: 2024.03.01
readtime: 10 min
---

# Article Title

Content goes here...
```

---

## Step 2: Create the HTML Wrapper (URL: `…/[section]/[slug]/`)
Use the `_template.html` file belonging to the section you are targeting.

```bash
mkdir -p [slug] && cp _template.html [slug]/index.html
```

This pattern applies to `tech-blog/`, `book-notes/`, and `courses/` (each uses its own `_template.html`).

---

## Step 3: Register the Post in `index.json`
Add a new entry to the relevant index.

```json
{
  "slug": "[slug]",        // Must match the filename (and the route slug)
  "title": "Article title",
  "category": "Category name",
  "date": "2024.03.01",
  "readtime": "10 min",
  "color": 1               // 1~6 (UI palette color for the book/article)
}
```

### Color Options (UI Palette)
- Color 1: Blue
- Color 2: Green
- Color 3: Red
- Color 4: Brown
- Color 5: Teal
- Color 6: Purple

---

## Add Another Article in the Same Category
If you are extending an existing `categories` entry, keep the wrapper creation the same as above, and update `index.json` accordingly.

```bash
mkdir -p [slug] && cp _template.html [slug]/index.html
```

```json
{
  "categories": {
    "DevOps": {
      "icon": "🐳",
      "posts": [
        {
          "slug": "docker-basics",
          "title": "Docker Getting Started",
          "category": "DOCKER",
          "date": "2024.03.20",
          "readtime": "10 min",
          "color": 5
        },
        {
          "slug": "kubernetes-intro",
          "title": "Kubernetes Introduction",
          "category": "K8S",
          "date": "2024.03.25",
          "readtime": "15 min",
          "color": 6
        }
      ]
    }
  }
}
```

---

## Local Server (Preview)
```bash
python3 -m http.server 8000
```

---

## Examples

### New course wrapper under `courses/`
```bash
mkdir -p courses/bayesian-inference && cp courses/_template.html courses/bayesian-inference/index.html
```

### New article wrapper under `tech-blog/`
```bash
mkdir -p tech-blog/my-post && cp tech-blog/_template.html tech-blog/my-post/index.html
```