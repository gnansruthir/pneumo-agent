// Global configuration and states
let predictionData = null;
let activeReports = null;

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initMouse3DTilt();
  initUpload();
  initStatsCounter();
  initReportTabs();
});

// 1. Particle field animation
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);
  
  const particles = [];
  const particleCount = 60;
  
  class Particle {
    constructor() {
      this.reset();
    }
    
    reset() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.size = Math.random() * 2 + 0.5;
      this.speedX = Math.random() * 0.4 - 0.2;
      this.speedY = Math.random() * -0.5 - 0.1; // Float upwards
      this.color = Math.random() > 0.5 ? 'rgba(0, 242, 254, 0.2)' : 'rgba(127, 0, 255, 0.2)';
    }
    
    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      
      // Reset if out of bounds
      if (this.y < 0 || this.x < 0 || this.x > width) {
        this.reset();
        this.y = height;
      }
    }
    
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.fill();
    }
  }
  
  for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
  }
  
  function animate() {
    ctx.clearRect(0, 0, width, height);
    particles.forEach((p) => {
      p.update();
      p.draw();
    });
    requestAnimationFrame(animate);
  }
  
  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });
  
  animate();
}

// 2. Mouse-tracking 3D Tilt for Hero Wireframe & Cards
function initMouse3DTilt() {
  const lungContainer = document.getElementById('lung-container');
  const cards = document.querySelectorAll('.pipeline-card');
  
  document.addEventListener('mousemove', (e) => {
    if (lungContainer) {
      const x = (window.innerWidth / 2 - e.clientX) / 30;
      const y = (window.innerHeight / 2 - e.clientY) / 30;
      lungContainer.style.transform = `rotateY(${x}deg) rotateX(${y}deg)`;
    }
  });

  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const xc = rect.width / 2;
      const yc = rect.height / 2;
      
      const angleX = (yc - y) / 10;
      const angleY = (x - xc) / 10;
      
      card.style.transform = `rotateX(${angleX}deg) rotateY(${angleY}deg) translateZ(10px)`;
    });
    
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'rotateX(0deg) rotateY(0deg) translateZ(0)';
    });
  });
}

// 3. Upload Zone and predict logic
function initUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const loader = document.getElementById('prediction-loader');
  
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  async function handleFile(file) {
    // Show loader, reset UI
    loader.classList.add('active');
    document.getElementById('results-container').style.display = 'none';
    document.getElementById('result-details').style.display = 'none';
    document.getElementById('report-panel').style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      if (data.success) {
        predictionData = data;
        displayResults(data);
      } else {
        alert("Prediction failed: " + (data.detail || "Unknown error"));
      }
    } catch (e) {
      console.error(e);
      alert("Error contacting prediction server.");
    } finally {
      loader.classList.remove('active');
    }
  }
}

// 4. Display Results
function displayResults(data) {
  // Toggle Visibility
  document.getElementById('results-container').style.display = 'grid';
  document.getElementById('result-details').style.display = 'block';
  
  // Set images
  document.getElementById('input-preview').src = data.input_image;
  document.getElementById('heatmap-preview').src = data.heatmap_image || data.input_image;
  
  // Set Triage
  const triageBadge = document.getElementById('triage-badge');
  triageBadge.className = 'triage-badge'; // reset
  triageBadge.innerText = data.triage;
  
  if (data.triage === 'Urgent') {
    triageBadge.classList.add('triage-urgent');
  } else if (data.triage === 'Follow-up') {
    triageBadge.classList.add('triage-followup');
  } else {
    triageBadge.classList.add('triage-clear');
  }

  // Populate findings list
  const findingsList = document.getElementById('findings-list');
  findingsList.innerHTML = '';

  // Sort classes: positives first, then by probability descending
  const sortedClasses = Object.keys(data.predictions).sort((a, b) => {
    const aPred = data.predictions[a];
    const bPred = data.predictions[b];
    if (aPred.detected !== bPred.detected) {
      return aPred.detected ? -1 : 1;
    }
    return bPred.probability - aPred.probability;
  });

  sortedClasses.forEach(name => {
    const info = data.predictions[name];
    const item = document.createElement('div');
    item.className = `finding-item ${info.detected ? 'detected' : 'not-detected'}`;
    
    // Check mark or circle
    const statusSymbol = info.detected ? '✦' : '○';
    const highlightClass = info.detected ? 'style="color: #00F2FE; font-weight: bold;"' : 'style="color: var(--text-muted);"';
    
    item.innerHTML = `
      <span ${highlightClass}>${statusSymbol} ${name}</span>
      <span class="finding-prob" ${highlightClass}>${(info.probability * 100).toFixed(1)}%</span>
    `;
    findingsList.appendChild(item);
  });

  // Setup Generate Report Click Listener
  const reportBtn = document.getElementById('btn-generate-report');
  reportBtn.onclick = generateReport;
}

// 5. Generate Report call
async function generateReport() {
  if (!predictionData) return;
  
  const reportPanel = document.getElementById('report-panel');
  const reportLoader = document.getElementById('report-loader');
  const reportBody = document.getElementById('report-body');

  reportPanel.style.display = 'block';
  reportLoader.classList.add('active');
  reportBody.innerHTML = '';
  
  // Scroll to report panel
  reportPanel.scrollIntoView({ behavior: 'smooth' });

  try {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        detected_findings: predictionData.detected_findings,
        triage: predictionData.triage,
        predictions: predictionData.predictions
      })
    });
    
    activeReports = await response.json();
    displayReportContent('clinical');
  } catch (e) {
    console.error(e);
    reportBody.innerHTML = '<p style="color: #ff3b30; font-family: var(--font-mono)">Failed to generate agentic reports.</p>';
  } finally {
    reportLoader.classList.remove('active');
  }
}

// 6. Handle tab clicks
function initReportTabs() {
  const tabs = document.querySelectorAll('.report-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      const type = tab.getAttribute('data-tab');
      displayReportContent(type);
    });
  });
}

function displayReportContent(tabType) {
  if (!activeReports) return;
  const reportBody = document.getElementById('report-body');
  
  let content = '';
  if (tabType === 'clinical') {
    content = activeReports.clinical_report;
  } else if (tabType === 'patient_en') {
    content = activeReports.patient_summary_en;
  } else if (tabType === 'patient_hi') {
    content = activeReports.patient_summary_hi;
  }
  
  // Convert basic markdown heading / list items to HTML for rendering
  reportBody.innerHTML = formatMarkdown(content);
}

// Basic regex helper to parse markdown sections
function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/^#\s+(.+)$/gm, '<h3 style="margin: 20px 0 10px; color: var(--accent-teal)">$1</h3>')
    .replace(/^##\s+(.+)$/gm, '<h4 style="margin: 15px 0 8px; color: #FFF; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;">$1</h4>')
    .replace(/^\-\s+\*\*(.+?)\*\*(.+)$/gm, '<div style="margin-bottom: 8px;"><strong>$1</strong>$2</div>')
    .replace(/^\-\s+(.+)$/gm, '<div style="margin-left: 10px; margin-bottom: 6px;">• $1</div>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code style="background: rgba(0,242,254,0.1); color: var(--accent-teal); padding: 2px 6px; border-radius: 4px;">$1</code>');
}

// 7. Stats counter up animation on scroll
function initStatsCounter() {
  const counters = document.querySelectorAll('.count-up');
  const speed = 200;
  
  const animate = (counter) => {
    const target = +counter.getAttribute('data-target');
    const count = +counter.innerText;
    const increment = target / speed;
    
    if (count < target) {
      counter.innerText = Math.ceil(count + increment);
      setTimeout(() => animate(counter), 1);
    } else {
      counter.innerText = target.toLocaleString();
    }
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animate(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(c => observer.observe(c));
}
