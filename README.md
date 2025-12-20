# Step 1: 마크다운 파일 작성
- `posts/` 폴더에 `[slug].md` 파일 생성

```markdown
---
title: 글 제목
category: 카테고리명
date: 2024.03.01
readtime: 10 min
---

# Step 2: 글 제목

내용 작성...
```

# Step 2: HTML 파일 생성
``` bash
cp _template.html [slug].html
```

# Step 3: index.json에 추가

``` json
{
  "slug": "[slug]",        // 파일명과 동일해야 함
  "title": "글 제목",
  "category": "카테고리명",
  "date": "2024.03.01",
  "readtime": "10 min",
  "color": 1              // 1~6 (책 색상)
}
```

# Color 옵션 (책 색상)
- 색상1: 파랑
- 색상2: 초록
- 색상 3: 빨강
- 색상 4: 갈색
- 색상 5: 청록
- 색상 6: 보라

---
# Add article in same category

``` bash
cp _template.html [slug].html
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

# Local Server Run

``` bash
python3 -m http.server 8000
```
