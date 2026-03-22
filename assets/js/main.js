/**
 * Library Blog - Main JavaScript
 * Handles loading posts, rendering shelves, search, and article display
 */

// ==================== EXECUTABLE CODE COUNTER ====================
let executableCodeCounter = 0;

// ==================== MARKDOWN PARSER ====================
marked.setOptions({
    langPrefix: 'language-' // ```cpp → <code class="language-cpp">
});

// Custom renderer for {run} tag support
const renderer = new marked.Renderer();

renderer.code = function(codeObj, language) {
    // marked.js 버전 호환: 객체 또는 문자열 처리
    let code, lang;
    if (typeof codeObj === 'object' && codeObj !== null) {
        // 최신 marked.js (v5+)
        code = codeObj.text || codeObj.raw || '';
        lang = codeObj.lang || '';
    } else {
        // 구버전 marked.js
        code = codeObj;
        lang = language || '';
    }
    
    // ```python {run} 형식 감지
    const isExecutable = lang && lang.includes('{run}');
    const actualLang = lang ? lang.replace('{run}', '').replace('{RUN}', '').trim() : '';
    
    if (isExecutable && actualLang === 'python') {
        executableCodeCounter++;
        const idx = executableCodeCounter;
        
        return `
        <div class="executable-code">
            <div class="code-header">
                <span class="lang-label">Python</span>
                <div class="code-buttons">
                    <button class="btn-toggle" id="toggle-btn-${idx}" onclick="toggleCode('wrapper-${idx}', 'toggle-btn-${idx}')">Show Code</button>
                    <button class="btn-toggle btn-output-toggle" id="output-toggle-btn-${idx}" onclick="toggleOutput('output-${idx}', 'output-toggle-btn-${idx}')" style="display:none;">Hide Output</button>
                    <button class="btn-run" onclick="runPython('code-${idx}', 'output-${idx}')">Run</button>
                </div>
            </div>
            <div class="code-wrapper collapsed" id="wrapper-${idx}">
                <pre><code id="code-${idx}" class="language-python">${escapeHtml(code)}</code></pre>
            </div>
            <div class="code-output" id="output-${idx}"></div>
        </div>`;
    }
    
    // 일반 코드 블록 (기존 방식)
    const langClass = actualLang ? `language-${actualLang}` : '';
    return `<pre><code class="${langClass}">${escapeHtml(code)}</code></pre>`;
};

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

marked.use({ renderer });

function parseMarkdownWithFrontmatter(text) {
    const frontmatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/;
    const match = text.match(frontmatterRegex);
    
    if (match) {
        const frontmatter = {};
        match[1].split('\n').forEach(line => {
            const colonIndex = line.indexOf(':');
            if (colonIndex > 0) {
                const key = line.slice(0, colonIndex).trim();
                let value = line.slice(colonIndex + 1).trim();
                if ((value.startsWith('"') && value.endsWith('"')) || 
                    (value.startsWith("'") && value.endsWith("'"))) {
                    value = value.slice(1, -1);
                }
                frontmatter[key] = value;
            }
        });
        return { frontmatter, content: match[2] };
    }
    return { frontmatter: {}, content: text };
}

// ==================== LOAD INDEX JSON ====================
async function loadIndex(path) {
    try {
        const response = await fetch(path);
        if (!response.ok) throw new Error('Not found');
        return await response.json();
    } catch (error) {
        console.error('Error loading index:', error);
        return { categories: {} };
    }
}

// ==================== RENDER TECH SHELVES ====================
function renderTechShelves(data, container) {
    container.innerHTML = '';
    
    Object.entries(data.categories).forEach(([category, categoryData]) => {
        const section = document.createElement('div');
        section.className = 'category-section';
        section.dataset.category = category;
        
        const booksHTML = categoryData.posts.map(post => `
            <a href="${post.slug}/" 
               class="book-front book-tech-${post.color || 1}" 
               data-title="${post.title.toLowerCase()}" 
               data-id="${post.slug}">
                <div class="book-front-cover">
                    <span class="book-front-category">${post.category || category}</span>
                    <h3 class="book-front-title">${post.title}</h3>
                    <div class="book-front-meta">
                        <span class="book-front-date">${post.date}</span>
                        <span class="book-front-time">${post.readtime || '5 min'}</span>
                    </div>
                </div>
            </a>
        `).join('');

        section.innerHTML = `
            <div class="category-header">
                <div class="category-icon">${categoryData.icon || '📁'}</div>
                <h3 class="category-title">${category}</h3>
                <span class="category-count">${categoryData.posts.length} ARTICLES</span>
            </div>
            <div class="bookshelf">
                <div class="shelf-top-frame"></div>
                <div class="shelf-row">
                    ${booksHTML}
                    <div class="shelf-plank"></div>
                </div>
                <div class="shelf-bottom-frame"></div>
            </div>
        `;
        container.appendChild(section);
    });
}

// ==================== RENDER BOOK SHELVES ====================
function renderBookShelves(data, container) {
    container.innerHTML = '';
    
    Object.entries(data.categories).forEach(([category, categoryData]) => {
        const books = categoryData.books;
        const midPoint = Math.ceil(books.length / 2);
        const firstRow = books.slice(0, midPoint);
        const secondRow = books.slice(midPoint);

        const section = document.createElement('div');
        section.className = 'category-section';
        section.dataset.category = category;

        const renderBook = (book) => `
            <a href="${book.slug}/" 
               class="book-with-cover" 
               data-title="${book.title.toLowerCase()}" 
               data-author="${(book.author || '').toLowerCase()}" 
               data-id="${book.slug}">
                <div class="book-cover">
                    <img src="${book.cover || 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=300&h=400&fit=crop'}" alt="${book.title}" loading="lazy">
                    <div class="book-cover-overlay">
                        <div class="book-cover-title">${book.title}</div>
                        <div class="book-cover-author">${book.author || ''}</div>
                    </div>
                </div>
            </a>
        `;

        section.innerHTML = `
            <div class="category-header">
                <div class="category-icon">${categoryData.icon || '📚'}</div>
                <h3 class="category-title">${category}</h3>
                <span class="category-count">${books.length} BOOKS</span>
            </div>
            <div class="bookshelf">
                <div class="shelf-top-frame"></div>
                <div class="shelf-rows-container">
                    <div class="shelf-row">
                        ${firstRow.map(renderBook).join('')}
                        <div class="shelf-plank"></div>
                    </div>
                    ${secondRow.length > 0 ? `
                    <div class="shelf-row">
                        ${secondRow.map(renderBook).join('')}
                        <div class="shelf-plank"></div>
                    </div>
                    ` : ''}
                </div>
                <div class="shelf-bottom-frame"></div>
            </div>
        `;
        container.appendChild(section);
    });
}

// ==================== RENDER COURSE SHELVES ====================
function renderCourseShelves(data, container) {
    container.innerHTML = '';
    
    Object.entries(data.categories).forEach(([category, categoryData]) => {
        const section = document.createElement('div');
        section.className = 'category-section';
        section.dataset.category = category;
        
        const coursesHTML = categoryData.courses.map(course => `
            <a href="${course.slug}/" 
               class="book-front book-tech-${course.color || 1}" 
               data-title="${course.title.toLowerCase()}" 
               data-id="${course.slug}">
                <div class="book-front-cover">
                    <span class="book-front-category">${course.category}</span>
                    <h3 class="book-front-title">${course.title}</h3>
                    <div class="book-front-meta">
                        <span class="book-front-date">${course.semester}</span>
                    </div>
                </div>
            </a>
        `).join('');

        section.innerHTML = `
            <div class="category-header">
                <div class="category-icon">${categoryData.icon || '📚'}</div>
                <h3 class="category-title">${category}</h3>
                <span class="category-count">${categoryData.courses.length} COURSES</span>
            </div>
            <div class="bookshelf">
                <div class="shelf-top-frame"></div>
                <div class="shelf-row">
                    ${coursesHTML}
                    <div class="shelf-plank"></div>
                </div>
                <div class="shelf-bottom-frame"></div>
            </div>
        `;
        container.appendChild(section);
    });
}

// ==================== SEARCH FUNCTIONALITY ====================
function setupSearch(inputId, clearId, containerId, resultsId, noResultsId, itemSelector) {
    const input = document.getElementById(inputId);
    const clearBtn = document.getElementById(clearId);
    const container = document.getElementById(containerId);
    const resultsCount = document.getElementById(resultsId);
    const noResults = document.getElementById(noResultsId);
    
    if (!input) return;

    input.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        clearBtn.classList.toggle('visible', query.length > 0);
        
        if (query === '') {
            container.querySelectorAll(itemSelector).forEach(item => item.classList.remove('hidden'));
            container.querySelectorAll('.category-section').forEach(section => section.classList.remove('hidden'));
            resultsCount.textContent = '';
            noResults.style.display = 'none';
            container.style.display = 'block';
            return;
        }

        let totalVisible = 0;
        
        container.querySelectorAll('.category-section').forEach(section => {
            let visibleInCategory = 0;
            section.querySelectorAll(itemSelector).forEach(item => {
                const title = item.dataset.title || '';
                const author = item.dataset.author || '';
                if (title.includes(query) || author.includes(query)) {
                    item.classList.remove('hidden');
                    visibleInCategory++;
                    totalVisible++;
                } else {
                    item.classList.add('hidden');
                }
            });
            section.classList.toggle('hidden', visibleInCategory === 0);
        });

        resultsCount.textContent = totalVisible > 0 ? `${totalVisible} results found` : '';
        noResults.style.display = totalVisible === 0 ? 'block' : 'none';
        container.style.display = totalVisible === 0 ? 'none' : 'block';
    });

    clearBtn.addEventListener('click', function() {
        input.value = '';
        input.dispatchEvent(new Event('input'));
    });
}

// ==================== LOAD ARTICLE ====================
async function loadArticle(slug, postsPath) {
    try {
      // slug가 "html/xxx" 형태일 수 있으므로 마지막 조각만 사용
      const mdSlug = slug.split('/').pop();
  
      const response = await fetch(`${postsPath}/${mdSlug}.md`);
      if (!response.ok) throw new Error('Article not found');
  
      const text = await response.text();
      return parseMarkdownWithFrontmatter(text);
    } catch (error) {
      console.error('Error loading article:', error);
      return null;
    }
  }
  

function addCopyButtons() {
    document.querySelectorAll('pre code').forEach(codeBlock => {
      const pre = codeBlock.parentElement;
  
      // 이미 버튼 있으면 중복 생성 방지
      if (pre.querySelector('.copy-code-button')) return;
      
      // executable-code 내부의 코드 블록은 제외 (이미 버튼 있음)
      if (pre.closest('.executable-code')) return;
  
      const button = document.createElement('button');
      button.className = 'copy-code-button';
      button.textContent = 'Copy';
  
      button.addEventListener('click', () => {
        navigator.clipboard.writeText(codeBlock.innerText);
  
        button.textContent = 'Copied!';
        button.classList.add('copied');
  
        setTimeout(() => {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 1200);
      });
  
      pre.appendChild(button);
    });
  }

// ==================== RENDER ARTICLE ====================
// function renderArticle(data, contentElement) {
//     // 카운터 리셋 (새 글 렌더링 시)
//     executableCodeCounter = 0;
    
//     const html = marked.parse(data.content);
//     contentElement.innerHTML = html;

//     // Prism 하이라이팅
//     if (window.Prism) {
//         Prism.highlightAllUnder(contentElement);
//     }
    
//     setTimeout(() => {
//         addCopyButtons();
//     }, 0);
    
//     // KaTeX
//     if (typeof renderMathInElement !== 'undefined') {
//         setTimeout(() => {
//             renderMathInElement(contentElement, {
//                 delimiters: [
//                     {left: '$$', right: '$$', display: true},
//                     {left: '$', right: '$', display: false},
//                     {left: '\\[', right: '\\]', display: true},
//                     {left: '\\(', right: '\\)', display: false}
//                 ],
//                 throwOnError: false
//             });
//         }, 50);
//     }
// }

// ==================== RENDER ARTICLE ====================
function renderArticle(data, contentElement) {
    // 카운터 리셋 (새 글 렌더링 시)
    executableCodeCounter = 0;
    
    const html = marked.parse(data.content);
    contentElement.innerHTML = html;

    // Prism 하이라이팅
    if (window.Prism) {
        Prism.highlightAllUnder(contentElement);
    }
    
    setTimeout(() => {
        addCopyButtons();
    }, 0);
    
    const sidebarElement = document.getElementById('article-sidebar');

    const finishArticle = () => {
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(contentElement, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false
            });
        }
        generateTOC(contentElement, sidebarElement);
    };

    requestAnimationFrame(finishArticle);
}


// ==================== INIT FUNCTIONS ====================

// Tech Blog List Page
async function initTechBlogList() {
    const container = document.getElementById('tech-shelves');
    if (!container) return;
    
    const data = await loadIndex('index.json');
    renderTechShelves(data, container);
    setupSearch('tech-search', 'tech-search-clear', 'tech-shelves', 'tech-results-count', 'tech-no-results', '.book-front');
}

// Book Notes List Page
async function initBookNotesList() {
    const container = document.getElementById('book-shelves');
    if (!container) return;
    
    const data = await loadIndex('index.json');
    renderBookShelves(data, container);
    setupSearch('book-search', 'book-search-clear', 'book-shelves', 'book-results-count', 'book-no-results', '.book-with-cover');
}

// Courses List Page
async function initCoursesList() {
    const container = document.getElementById('course-shelves');
    if (!container) return;
    
    const data = await loadIndex('index.json');
    renderCourseShelves(data, container);
    setupSearch('course-search', 'course-search-clear', 'course-shelves', 'course-results-count', 'course-no-results', '.book-front');
}

// Article Page (Tech Blog)
async function initTechArticle(slug) {
    const contentElement = document.getElementById('article-content');
    if (!contentElement) return;
    
    const data = await loadArticle(slug, '../posts');
    
    if (data) {
        if (data.frontmatter.title) {
            document.getElementById('article-title').textContent = data.frontmatter.title;
            document.title = `${data.frontmatter.title} | Tech Blog`;
        }
        if (data.frontmatter.category) {
            document.getElementById('article-category').textContent = data.frontmatter.category;
        }
        if (data.frontmatter.date) {
            document.getElementById('article-date').textContent = data.frontmatter.date;
        }
        if (data.frontmatter.readtime) {
            document.getElementById('article-readtime').textContent = data.frontmatter.readtime + ' read';
        }
        
        renderArticle(data, contentElement);
    } else {
        contentElement.innerHTML = '<p>Article not found.</p>';
    }
}

// Article Page (Book Notes)
async function initBookArticle(slug) {
    const contentElement = document.getElementById('article-content');
    if (!contentElement) return;
    
    const data = await loadArticle(slug, '../posts');
    
    if (data) {
        if (data.frontmatter.title) {
            document.getElementById('article-title').textContent = data.frontmatter.title;
            document.title = `${data.frontmatter.title} | Book Notes`;
        }
        if (data.frontmatter.category) {
            document.getElementById('article-category').textContent = data.frontmatter.category;
        }
        if (data.frontmatter.date) {
            document.getElementById('article-date').textContent = 'Read: ' + data.frontmatter.date;
        }
        if (data.frontmatter.rating) {
            document.getElementById('article-readtime').textContent = data.frontmatter.rating;
        }
        if (data.frontmatter.author) {
            const authorEl = document.getElementById('article-author');
            if (authorEl) {
                authorEl.textContent = data.frontmatter.author;
                authorEl.style.display = 'block';
            }
        }
        if (data.frontmatter.cover) {
            const coverWrapper = document.getElementById('article-cover-wrapper');
            const coverImg = document.getElementById('article-cover');
            if (coverWrapper && coverImg) {
                coverImg.src = data.frontmatter.cover;
                coverWrapper.style.display = 'block';
            }
        }
        
        renderArticle(data, contentElement);
    } else {
        contentElement.innerHTML = '<p>Book note not found.</p>';
    }
}

// Article Page (Courses)
async function initCourseArticle(slug) {
    const contentElement = document.getElementById('article-content');
    if (!contentElement) return;
    
    const data = await loadArticle(slug, '../posts');

    if (data) {
        if (data.frontmatter.title) {
            document.getElementById('article-title').textContent = data.frontmatter.title;
            document.title = `${data.frontmatter.title} | Courses`;
        }
        if (data.frontmatter.category) {
            document.getElementById('article-category').textContent = data.frontmatter.category;
        }
        if (data.frontmatter.semester) {
            document.getElementById('article-date').textContent = data.frontmatter.semester;
        }
        
        renderArticle(data, contentElement);
    } else {
        contentElement.innerHTML = '<p>Course not found.</p>';
    }
}
// ==================== GENERATE TABLE OF CONTENTS ====================
function generateTOC(contentElement, sidebarElement) {
    if (!contentElement || !sidebarElement) return;

    const headings = Array.from(contentElement.querySelectorAll('h1, h2'));
    if (headings.length === 0) {
        sidebarElement.style.display = 'none';
        return;
    }

    sidebarElement.style.display = '';

    headings.forEach((heading, index) => {
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
    });

    let tocHTML = '<p class="sidebar-title">Contents</p><ul class="sidebar-nav">';

    headings.forEach((heading) => {
        const level = heading.tagName.toLowerCase();
        const id = heading.id;
        const labelHtml = heading.innerHTML.trim();
        tocHTML += `<li><a href="#${id}" class="toc-${level}">${labelHtml}</a></li>`;
    });

    tocHTML += '</ul>';
    sidebarElement.innerHTML = tocHTML;

    const tocLinks = sidebarElement.querySelectorAll('a');

    const scrollOffset = 140;
    let scrollTicking = false;

    function syncTocWithScroll() {
        if (scrollTicking) return;
        scrollTicking = true;
        requestAnimationFrame(() => {
            scrollTicking = false;
            let current = headings[0];
            for (const h of headings) {
                if (h.getBoundingClientRect().top <= scrollOffset) {
                    current = h;
                }
            }
            tocLinks.forEach((link) => link.classList.remove('active'));
            const activeLink = sidebarElement.querySelector(
                `a[href="#${CSS.escape(current.id)}"]`
            );
            if (activeLink) {
                activeLink.classList.add('active');
                activeLink.scrollIntoView({
                    block: 'nearest',
                    inline: 'nearest',
                    behavior: 'auto'
                });
            }
        });
    }

    window.addEventListener('scroll', syncTocWithScroll, { passive: true });
    syncTocWithScroll();

    tocLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// Export for global use
window.LibraryBlog = {
    initTechBlogList,
    initBookNotesList,
    initCoursesList,
    initTechArticle,
    initBookArticle,
    initCourseArticle,
    loadIndex,
    renderTechShelves,
    renderBookShelves,
    renderCourseShelves,
    generateTOC
};