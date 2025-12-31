// Pyodide Runner with Package Support
let pyodideInstance = null;
let pyodideLoading = false;
let loadedPackages = new Set();

async function loadPyodideIfNeeded() {
    if (pyodideInstance) return pyodideInstance;
    if (pyodideLoading) {
        while (pyodideLoading) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return pyodideInstance;
    }
    
    pyodideLoading = true;
    const statusEl = document.getElementById('pyodide-status');
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = 'Loading Python...';
    }
    
    try {
        pyodideInstance = await loadPyodide();
        if (statusEl) {
            statusEl.textContent = 'Python Ready';
            setTimeout(() => { statusEl.style.display = 'none'; }, 2000);
        }
    } catch (error) {
        if (statusEl) {
            statusEl.textContent = 'Python Load Failed';
            statusEl.style.background = '#c62828';
        }
        console.error('Pyodide load error:', error);
    }
    
    pyodideLoading = false;
    return pyodideInstance;
}

// 코드에서 필요한 패키지 감지
function detectPackages(code) {
    const packages = [];
    const importRegex = /^\s*(?:import|from)\s+(\w+)/gm;
    let match;
    
    while ((match = importRegex.exec(code)) !== null) {
        const pkg = match[1];
        // Pyodide에서 지원하는 주요 패키지 매핑
        const packageMap = {
            'numpy': 'numpy',
            'np': 'numpy',
            'pandas': 'pandas',
            'pd': 'pandas',
            'matplotlib': 'matplotlib',
            'plt': 'matplotlib',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'sympy': 'sympy',
            'networkx': 'networkx'
        };
        
        if (packageMap[pkg] && !loadedPackages.has(packageMap[pkg])) {
            packages.push(packageMap[pkg]);
        }
    }
    
    return [...new Set(packages)];
}

async function runPython(codeBlockId, outputId) {
    const codeEl = document.getElementById(codeBlockId);
    const outputEl = document.getElementById(outputId);
    const runBtn = document.querySelector(`[onclick="runPython('${codeBlockId}', '${outputId}')"]`);
    
    if (!codeEl || !outputEl) return;
    
    const code = codeEl.textContent;
    
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = 'Running...';
    }
    
    outputEl.style.display = 'block';
    outputEl.innerHTML = '<span class="loading">Running...</span>';
    outputEl.className = 'code-output';
    
    try {
        const pyodide = await loadPyodideIfNeeded();
        
        if (!pyodide) {
            throw new Error('Failed to load Python. Please refresh the page.');
        }
        
        // 필요한 패키지 감지 및 설치
        const neededPackages = detectPackages(code);
        if (neededPackages.length > 0) {
            outputEl.innerHTML = `<span class="loading">Installing ${neededPackages.join(', ')}...</span>`;
            await pyodide.loadPackage(neededPackages);
            neededPackages.forEach(pkg => loadedPackages.add(pkg));
        }
        
        // matplotlib 설정 (그래프를 이미지로 출력 - 다크 테마)
        if (code.includes('matplotlib') || code.includes('plt')) {
            pyodide.runPython(`
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import io, base64

# Dark theme styling
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#1a1a1a',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#b0b0b0',
    'text.color': '#cccccc',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'grid.color': '#2a2a2a',
    'grid.alpha': 0.5,
    'lines.color': '#c9a961',
    'axes.prop_cycle': plt.cycler(color=['#c9a961', '#6b9ac4', '#97c4a0', '#d4a5a5', '#a5a5d4', '#d4c9a5'])
})

def _show_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                facecolor='#1a1a1a', edgecolor='none', pad_inches=0.2)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close('all')
    return img_base64

# Override plt.show
plt.show = lambda: None
`);
        }
        
        // stdout/stderr 캡처 설정
        pyodide.runPython(`
import sys
from io import StringIO
_stdout_capture = StringIO()
_stderr_capture = StringIO()
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture
`);
        
        // 코드 실행
        let result;
        try {
            result = pyodide.runPython(code);
        } catch (pyError) {
            throw pyError;
        }
        
        // stdout/stderr 가져오기
        const stdout = pyodide.runPython('_stdout_capture.getvalue()');
        const stderr = pyodide.runPython('_stderr_capture.getvalue()');
        
        // 그래프가 있으면 이미지로 변환
        let plotImg = '';
        if (code.includes('matplotlib') || code.includes('plt')) {
            try {
                const imgBase64 = pyodide.runPython('_show_plot()');
                if (imgBase64) {
                    plotImg = `<img src="data:image/png;base64,${imgBase64}" class="plot-image" onclick="openImageModal(this.src)" title="Click to enlarge">`;
                }
            } catch (e) {
                // 그래프 없으면 무시
            }
        }
        
        // 결과 조합
        let output = '';
        if (stdout) output += escapeHtml(stdout);
        if (stderr) output += (output ? '\n' : '') + escapeHtml(stderr);
        if (result !== undefined && result !== null) {
            const resultStr = String(result);
            if (resultStr !== 'None' && resultStr !== '') {
                output += (output ? '\n' : '') + escapeHtml(resultStr);
            }
        }
        
        let finalHtml = '';
        if (output) finalHtml += '<pre>' + output + '</pre>';
        if (plotImg) finalHtml += plotImg;
        if (!finalHtml) finalHtml = '<pre>(No output)</pre>';
        
        outputEl.innerHTML = finalHtml;
        outputEl.classList.add('success');
        
        // 그래프가 있으면 출력 영역 확장
        if (plotImg) {
            outputEl.classList.add('has-plot');
        }
        
        // 출력 토글 버튼 표시
        const outputToggleBtn = document.getElementById('output-toggle-btn-' + outputId.replace('output-', ''));
        if (outputToggleBtn) {
            outputToggleBtn.style.display = 'inline-block';
        }
        
    } catch (error) {
        outputEl.innerHTML = '<pre class="error-text">' + escapeHtml(error.message) + '</pre>';
        outputEl.classList.add('error');
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.textContent = 'Run';
        }
    }
}

function toggleCode(wrapperId, btnId) {
    const wrapper = document.getElementById(wrapperId);
    const btn = document.getElementById(btnId);
    if (wrapper) {
        wrapper.classList.toggle('collapsed');
        if (btn) {
            btn.textContent = wrapper.classList.contains('collapsed') ? 'Show Code' : 'Hide Code';
        }
    }
}

function toggleOutput(outputId, btnId) {
    const output = document.getElementById(outputId);
    const btn = document.getElementById(btnId);
    if (output) {
        output.classList.toggle('collapsed');
        if (btn) {
            btn.textContent = output.classList.contains('collapsed') ? 'Show Output' : 'Hide Output';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}