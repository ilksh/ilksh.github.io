/**
 * Library Blog - Main JavaScript
 * Handles loading posts, rendering shelves, search, and article display
 */

// ==================== MARKDOWN PARSER ====================
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
            <a href="${post.slug}.html" 
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
            <a href="${book.slug}.html" 
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
            <a href="${course.slug}.html" 
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
        const response = await fetch(`${postsPath}/${slug}.md`);
        if (!response.ok) throw new Error('Article not found');
        const text = await response.text();
        return parseMarkdownWithFrontmatter(text);
    } catch (error) {
        console.error('Error loading article:', error);
        return null;
    }
}

// ==================== RENDER ARTICLE ====================
function renderArticle(data, contentElement) {
    const html = marked.parse(data.content);
    contentElement.innerHTML = html;
    
    // Render KaTeX math if available
    if (typeof renderMathInElement !== 'undefined') {
        setTimeout(() => {
            renderMathInElement(contentElement, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false
            });
        }, 50);
    }
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
    
    const data = await loadArticle(slug, 'posts');
    
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
    
    const data = await loadArticle(slug, 'posts');
    
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
    
    const data = await loadArticle(slug, 'posts');
    
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
    renderCourseShelves
};