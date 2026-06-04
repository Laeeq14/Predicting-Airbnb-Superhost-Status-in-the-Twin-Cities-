/* ===================================================================
   Superhost Predictor — Frontend Application
   Chart.js + Leaflet.js + FastAPI endpoints
   =================================================================== */

'use strict';

// ─────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────
let metadata = null;
let neighStats = {};        // module-level cache for neighbourhood stats
let gaugeChart   = null;
let radarChart   = null;
let scaleChart   = null;
let importChart  = null;
let leafletMap   = null;
let predictTimer = null;
let scaleTimer   = null;

const SLIDER_IDS = [
  'review_scores_rating', 'reviews_per_month', 'host_response_rate',
  'host_acceptance_rate', 'host_experience_years', 'host_listings_count', 'num_amenities'
];

// ─────────────────────────────────────────────────────────────────────
// Colour helpers — green / amber / red palette for light theme
// ─────────────────────────────────────────────────────────────────────
function probColor(p) {
  if (p >= 0.70) return '#1a7a4a';
  if (p >= 0.55) return '#2d8f5e';
  if (p >= 0.40) return '#5cba85';
  if (p >= 0.25) return '#d97706';
  return '#dc2626';
}
function shRateColor(r) {
  if (r >= 0.70) return '#1a7a4a';
  if (r >= 0.55) return '#2d8f5e';
  if (r >= 0.40) return '#5cba85';
  if (r >= 0.25) return '#d97706';
  return '#dc2626';
}
function hexToRgba(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ─────────────────────────────────────────────────────────────────────
// Gauge (half-doughnut via Chart.js)
// ─────────────────────────────────────────────────────────────────────
function initGauge() {
  const ctx = document.getElementById('gaugeCanvas').getContext('2d');
  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: ['#2d8f5e', '#e8f5ee'],
        borderWidth: 0,
        borderRadius: 4,
      }]
    },
    options: {
      circumference: 180,
      rotation: 270,
      cutout: '72%',
      animation: { duration: 700, easing: 'easeInOutQuart' },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    }
  });
}

function updateGauge(probability) {
  const pct = Math.round(probability * 100);
  const col = probColor(probability);
  gaugeChart.data.datasets[0].data = [pct, 100 - pct];
  gaugeChart.data.datasets[0].backgroundColor = [col, '#e8f5ee'];
  gaugeChart.update('active');

  const gaugeEl = document.getElementById('gauge-pct');
  gaugeEl.textContent = pct + '%';
  gaugeEl.style.color = col;
  gaugeEl.style.background = '';
  gaugeEl.style.webkitTextFillColor = '';

  document.getElementById('confidence-bar').style.width = pct + '%';
  document.getElementById('confidence-bar').style.background =
    `linear-gradient(90deg, ${col}, ${hexToRgba(col, 0.6)})`;

  const verdict = document.getElementById('prediction-verdict');
  if (probability >= 0.5) {
    verdict.textContent = 'Superhost';
    verdict.className = 'prediction-verdict superhost';
  } else {
    verdict.textContent = 'Not Yet Superhost';
    verdict.className = 'prediction-verdict not-yet';
  }
}

// ─────────────────────────────────────────────────────────────────────
// Radar Chart
// ─────────────────────────────────────────────────────────────────────
function normalise(val, minV, maxV) {
  return Math.min(1, Math.max(0, (val - minV) / (maxV - minV)));
}

const RADAR_RANGES = {
  review_scores_rating:  [1,   5],
  reviews_per_month:     [0,   10],
  host_response_rate:    [0,   100],
  host_acceptance_rate:  [0,   100],
  host_experience_years: [0,   20],
  host_listings_count:   [1,   50],
  num_amenities:         [0,   80],
};
const RADAR_LABELS = {
  review_scores_rating:  'Review Score',
  reviews_per_month:     'Reviews/mo',
  host_response_rate:    'Response Rate',
  host_acceptance_rate:  'Acceptance',
  host_experience_years: 'Experience',
  host_listings_count:   'Listings',
  num_amenities:         'Amenities',
};

function initRadar() {
  const ctx = document.getElementById('radarChart').getContext('2d');
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: SLIDER_IDS.map(id => RADAR_LABELS[id]),
      datasets: [
        {
          label: 'You',
          data: Array(7).fill(0),
          borderColor: '#2d8f5e',
          backgroundColor: 'rgba(45,143,94,0.12)',
          borderWidth: 2,
          pointBackgroundColor: '#2d8f5e',
          pointRadius: 4,
          tension: 0.3,
        },
        {
          label: 'Superhost Median',
          data: Array(7).fill(0),
          borderColor: '#5cba85',
          backgroundColor: 'rgba(92,186,133,0.07)',
          borderWidth: 2,
          borderDash: [5,3],
          pointBackgroundColor: '#5cba85',
          pointRadius: 4,
          tension: 0.3,
        },
      ]
    },
    options: {
      animation: { duration: 500 },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false, stepSize: 0.25 },
          grid:  { color: 'rgba(0,0,0,0.06)' },
          pointLabels: { color: '#4a6358', font: { size: 11, family: 'Inter' } },
          angleLines: { color: 'rgba(0,0,0,0.08)' },
        }
      },
      plugins: {
        legend: {
          labels: { color: '#4a6358', font: { size: 11, family: 'Inter' }, boxWidth: 12, padding: 14 }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const feat = SLIDER_IDS[ctx.dataIndex];
              const [mn, mx] = RADAR_RANGES[feat];
              const actual = ctx.raw * (mx - mn) + mn;
              return ` ${ctx.dataset.label}: ${actual.toFixed(1)}`;
            }
          }
        }
      }
    }
  });
}

function updateRadar(sliderVals) {
  if (!metadata) return;

  // Determine which benchmark to use: local county or metro-wide
  const select = document.getElementById('neighbourhood-select');
  const selectedCounty = select ? select.value : '__metro__';
  const warnEl = document.getElementById('neighbourhood-sample-warn');
  const labelEl = document.getElementById('radar-benchmark-label');

  const shAvgGlobal = metadata.superhost_avg || {};
  let benchmarkAvg = { ...shAvgGlobal };  // start with global
  let usingLocal = false;

  if (selectedCounty !== '__metro__' && neighStats[selectedCounty]) {
    const ns = neighStats[selectedCounty];
    const MIN_SAMPLE = 30;

    if (ns.listing_count >= MIN_SAMPLE) {
      // Blend: use local data for the axes we have, fall back for the rest
      benchmarkAvg = {
        ...shAvgGlobal,                                // fallback baseline
        review_scores_rating:  ns.median_review_score   ?? shAvgGlobal.review_scores_rating,
        host_response_rate:    ns.median_response_rate  ?? shAvgGlobal.host_response_rate,
        reviews_per_month:     ns.median_reviews_pm     ?? shAvgGlobal.reviews_per_month,
        // acceptance_rate, experience_years, listings, amenities: stay global
      };
      usingLocal = true;
      if (warnEl) { warnEl.hidden = true; warnEl.textContent = ''; }
      if (labelEl) labelEl.textContent = `Superhost Median · ${selectedCounty}`;
    } else {
      // Sample too small — use metro, show pill warning
      if (warnEl) {
        warnEl.hidden = false;
        warnEl.textContent = `⚠ N=${ns.listing_count} · Showing metro median`;
      }
      if (labelEl) labelEl.textContent = 'Superhost Median (metro)';
    }
  } else {
    if (warnEl) { warnEl.hidden = true; warnEl.textContent = ''; }
    if (labelEl) labelEl.textContent = 'Superhost Median';
  }

  const userNorm = SLIDER_IDS.map(id => {
    const [mn, mx] = RADAR_RANGES[id];
    return normalise(sliderVals[id] || 0, mn, mx);
  });
  const shNorm = SLIDER_IDS.map(id => {
    const [mn, mx] = RADAR_RANGES[id];
    return normalise(benchmarkAvg[id] || 0, mn, mx);
  });

  // Update legend label in chart dataset
  const benchLabel = labelEl ? labelEl.textContent : 'Superhost Median';
  radarChart.data.datasets[1].label = benchLabel;

  radarChart.data.datasets[0].data = userNorm;
  radarChart.data.datasets[1].data = shNorm;
  radarChart.update('active');
}

// ─────────────────────────────────────────────────────────────────────
// Scale Warning Chart
// ─────────────────────────────────────────────────────────────────────
function initScaleChart() {
  const ctx = document.getElementById('scaleChart').getContext('2d');
  scaleChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Superhost Probability',
        data: [],
        borderColor: '#2d8f5e',
        backgroundColor: (ctx) => {
          const chart = ctx.chart;
          const {ctx: c, chartArea} = chart;
          if (!chartArea) return 'rgba(45,143,94,0.08)';
          const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, 'rgba(45,143,94,0.18)');
          gradient.addColorStop(1, 'rgba(45,143,94,0.01)');
          return gradient;
        },
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#2d8f5e',
      }]
    },
    options: {
      animation: { duration: 600 },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          title: { display: true, text: 'Number of Listings', color: '#7d9b8a', font: { size: 11 } },
          grid:  { color: 'rgba(0,0,0,0.04)' },
          ticks: { color: '#7d9b8a', maxTicksLimit: 10 },
          border: { color: '#e2ece6' },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'Probability', color: '#7d9b8a', font: { size: 11 } },
          grid:  { color: 'rgba(0,0,0,0.04)' },
          ticks: { color: '#7d9b8a', callback: v => (v*100).toFixed(0)+'%' },
          border: { color: '#e2ece6' },
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.98)',
          borderColor: '#e2ece6',
          borderWidth: 1,
          titleColor: '#1a2e22',
          bodyColor: '#4a6358',
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          callbacks: {
            title: items => `${items[0].label} Listings`,
            label: item => ` Probability: ${(item.raw*100).toFixed(1)}%`,
          }
        },
        annotation: {}
      }
    }
  });
}

function updateScaleChart(curve, sweetSpot, currentListings) {
  scaleChart.data.labels = curve.map(d => d.listings);
  scaleChart.data.datasets[0].data = curve.map(d => d.probability);
  scaleChart.update('active');

  document.getElementById('sweet-spot-label').textContent =
    `Optimal Scale: ${sweetSpot} Listing${sweetSpot > 1 ? 's' : ''}`;
}

// ─────────────────────────────────────────────────────────────────────
// Feature Importance Chart
// ─────────────────────────────────────────────────────────────────────
function renderImportanceChart(features) {
  const top = features.slice(0, 15);
  const labels = top.map(f => {
    return f.feature
      .replace(/_log1p$/, ' (log)')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  });
  const vals   = top.map(f => f.importance);
  const maxVal = Math.max(...vals);

  const ctx = document.getElementById('importanceChart').getContext('2d');
  importChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance',
        data: vals,
        backgroundColor: vals.map(v => {
          const ratio = v / maxVal;
          // Green gradient: light to dark
          const r = Math.round(92  + (26-92)*ratio);
          const g = Math.round(186 + (122-186)*ratio);
          const b = Math.round(133 + (74-133)*ratio);
          return `rgba(${r},${g},${b},0.85)`;
        }),
        borderColor: 'transparent',
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      animation: { duration: 800 },
      scales: {
        x: {
          grid:  { color: 'rgba(0,0,0,0.04)' },
          ticks: { color: '#7d9b8a', font: { size: 10 } },
          border: { color: '#e2ece6' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#4a6358', font: { size: 10.5 } },
          border: { display: false },
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.98)',
          borderColor: '#e2ece6',
          borderWidth: 1,
          titleColor: '#1a2e22',
          bodyColor: '#4a6358',
          callbacks: {
            label: item => ` Importance: ${item.raw.toFixed(4)}`
          }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────
// Model Performance Table
// ─────────────────────────────────────────────────────────────────────
function renderPerfTable(perf) {
  const DISPLAY_ORDER = [
    'Logistic Regression', 'Decision Tree',
    'Random Forest', 'Random Forest (Tuned)',
    'XGBoost', 'XGBoost (Tuned)',
    'LightGBM', 'LightGBM (Tuned)',
    'CatBoost', 'CatBoost (Tuned)',
    'Voting Ensemble',
  ];
  const rows = DISPLAY_ORDER.filter(n => perf[n]);
  // Best = highest ROC-AUC (dynamic, from metadata or computed here)
  const bestName = metadata?.best_model_name
    || rows.reduce((best, n) => perf[n].roc_auc > (perf[best]?.roc_auc || 0) ? n : best, rows[0]);

  const tbody = document.getElementById('perf-tbody');
  tbody.innerHTML = rows.map(name => {
    const m = perf[name];
    const isBest = name === bestName;
    return `<tr class="${isBest ? 'best-model' : ''}">
      <td>${name}${isBest ? '<span class="best-badge">Best</span>' : ''}</td>
      <td>${(m.accuracy*100).toFixed(1)}%</td>
      <td class="metric-auc">${m.roc_auc.toFixed(4)}</td>
      <td>${(m.f1*100).toFixed(1)}%</td>
    </tr>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────────────
// Recommendations renderer
// ─────────────────────────────────────────────────────────────────────
function renderRecommendations(recs) {
  const el = document.getElementById('recommendations');
  if (!recs || recs.length === 0) {
    el.innerHTML = `<div class="no-recs card">
      <span class="trophy-icon">&#10003;</span>
      <h4>You're already at Superhost level!</h4>
      <p>Your metrics match or exceed the Superhost median on all key dimensions.<br/>
         Focus on maintaining consistency to retain your status.</p>
    </div>`;
    return;
  }
  el.innerHTML = recs.map((r, i) => `
    <div class="rec-card animate-fade-in" style="animation-delay:${i*0.1}s">
      <div class="rec-delta">+${(r.delta_probability*100).toFixed(0)}%</div>
      <div class="rec-label">${r.label}</div>
      <div class="rec-msg">${markdownBold(r.message)}</div>
      <div class="rec-footer">
        <span>Current: <strong>${formatVal(r.feature, r.current)}</strong></span>
        <span>Target: <strong>${formatVal(r.feature, r.target)}</strong></span>
      </div>
    </div>`).join('');
}

function markdownBold(text) {
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}
function formatVal(feat, val) {
  if (feat === 'host_response_rate' || feat === 'host_acceptance_rate') return val.toFixed(0) + '%';
  if (feat === 'review_scores_rating') return val.toFixed(1) + '/ 5.0';
  if (feat === 'reviews_per_month') return val.toFixed(1) + '/mo';
  return val.toFixed(0);
}

// ─────────────────────────────────────────────────────────────────────
// Map — with robust tile loading and light theme
// ─────────────────────────────────────────────────────────────────────
async function initMap(neighStats) {
  // Remove loading message
  const loadingMsg = document.getElementById('map-loading-msg');
  if (loadingMsg) loadingMsg.remove();

  // Initialise map
  leafletMap = L.map('map', {
    center: [45.06, -93.45],
    zoom: 8,
    zoomControl: true,
    attributionControl: true,
    preferCanvas: false,
  });

  // Use OpenStreetMap (light, reliable, no API key required)
  const osmLight = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
      crossOrigin: true,
    }
  );

  // CartoDB Positron as preferred light tile (matches InsideAirbnb style)
  const cartoLight = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
      crossOrigin: true,
    }
  );

  // Try CartoDB Positron first; fall back to OSM on error
  let cartoFailed = false;

  cartoLight.on('tileerror', function() {
    if (!cartoFailed) {
      cartoFailed = true;
      console.warn('CartoDB Positron tiles unavailable, switching to OSM');
      leafletMap.removeLayer(cartoLight);
      osmLight.addTo(leafletMap);
    }
  });

  cartoLight.addTo(leafletMap);

  // Also set a timeout: if no tile loaded in 6s, switch to OSM
  let tileLoaded = false;
  cartoLight.once('tileload', () => { tileLoaded = true; });
  setTimeout(() => {
    if (!tileLoaded && !cartoFailed) {
      console.warn('CartoDB tiles not responding, switching to OSM');
      cartoFailed = true;
      leafletMap.removeLayer(cartoLight);
      osmLight.addTo(leafletMap);
    }
  }, 6000);

  // Load GeoJSON from backend
  let geoData;
  try {
    const res = await fetch('/geojson');
    if (!res.ok) throw new Error('GeoJSON fetch failed: ' + res.status);
    geoData = await res.json();
  } catch(e) {
    console.warn('GeoJSON not available', e);
    document.getElementById('map').innerHTML =
        '<div class="hero-stat"><span class="dot"></span><span id="stat-model-name">Best Model</span> &middot; <strong id="stat-auc">—</strong> AUC</div>';
    return;
  }

  const geoLayer = L.geoJSON(geoData, {
    style: feature => {
      const name = feature.properties.neighbourhood || feature.properties.name || '';
      const stats = neighStats[name] || {};
      const rate = stats.superhost_rate || 0;
      return {
        fillColor: shRateColor(rate),
        fillOpacity: 0.5,
        color: '#ffffff',
        weight: 1.5,
        opacity: 0.8,
      };
    },
    onEachFeature: (feature, layer) => {
      const name = feature.properties.neighbourhood || feature.properties.name || 'Unknown';
      const s = neighStats[name] || {};
      const rate = s.superhost_rate ? (s.superhost_rate*100).toFixed(1) : '—';
      const col  = s.superhost_rate ? shRateColor(s.superhost_rate) : '#7d9b8a';
      layer.bindPopup(`
        <div class="popup-title">${name} County</div>
        <div class="popup-row">
          <span class="popup-key">Superhost Rate</span>
          <span class="sh-badge" style="background:${hexToRgba(col,0.12)};color:${col};border:1px solid ${hexToRgba(col,0.3)}">${rate}%</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Listings</span>
          <span class="popup-val">${s.listing_count !== undefined ? s.listing_count.toLocaleString() : '—'}</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Median Price</span>
          <span class="popup-val">$${s.median_price ? s.median_price.toFixed(0) : '—'}/night</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Review Score</span>
          <span class="popup-val">${s.median_review_score || '—'} / 5.0</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Response Rate</span>
          <span class="popup-val">${s.median_response_rate !== undefined ? s.median_response_rate.toFixed(0)+'%' : '—'}</span>
        </div>
        <div class="popup-row">
          <span class="popup-key">Reviews/mo</span>
          <span class="popup-val">${s.median_reviews_pm || '—'}</span>
        </div>
      `, { maxWidth: 270, className: 'custom-popup' });

      layer.on('mouseover', () => layer.setStyle({ fillOpacity: 0.75, weight: 2.5, opacity: 1 }));
      layer.on('mouseout',  () => layer.setStyle({ fillOpacity: 0.5, weight: 1.5, opacity: 0.8 }));
      layer.on('click', () => layer.openPopup());
    }
  }).addTo(leafletMap);

  // Fit map to GeoJSON bounds
  try {
    leafletMap.fitBounds(geoLayer.getBounds(), { padding: [20, 20] });
  } catch(e) {
    console.warn('fitBounds failed:', e);
  }
}

// ─────────────────────────────────────────────────────────────────────
// Slider helpers
// ─────────────────────────────────────────────────────────────────────
function getSliderValues() {
  const vals = {};
  SLIDER_IDS.forEach(id => {
    vals[id] = parseFloat(document.getElementById(id).value);
  });
  return vals;
}

function updateSliderDisplay(id, value) {
  const el = document.getElementById(`val-${id}`);
  if (!el) return;
  const formatted = id.includes('rate') ? value.toFixed(0)
    : id.includes('years') ? value.toFixed(1)
    : id === 'review_scores_rating' ? value.toFixed(1)
    : id === 'reviews_per_month' ? value.toFixed(1)
    : value.toFixed(0);

  // Keep the unit span
  const unit = el.querySelector('.slider-unit');
  el.innerHTML = formatted;
  if (unit) el.appendChild(unit);
  else el.insertAdjacentHTML('beforeend', '<span class="slider-unit"></span>');

  // Reattach unit text
  const units = { review_scores_rating:'/ 5.0', reviews_per_month:'/mo',
                  host_response_rate:'%', host_acceptance_rate:'%',
                  host_experience_years:'yrs', host_listings_count:'', num_amenities:'' };
  el.querySelector('.slider-unit').textContent = units[id] || '';

  // Update slider gradient fill
  const input = document.getElementById(id);
  const pct = ((value - parseFloat(input.min)) / (parseFloat(input.max) - parseFloat(input.min))) * 100;
  input.style.background = `linear-gradient(90deg, rgba(45,143,94,0.75) ${pct}%, #d5f0e0 ${pct}%)`;
}

// ─────────────────────────────────────────────────────────────────────
// API calls
// ─────────────────────────────────────────────────────────────────────
async function fetchPrediction(vals) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(vals),
  });
  return res.json();
}

async function fetchSimulation(vals) {
  const body = {...vals};
  delete body.host_listings_count;   // simulate will vary this
  body.max_listings = 50;
  const res = await fetch('/simulate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────
// Hard Threshold Checklist
// ─────────────────────────────────────────────────────────────────────
const HARD_RULES = [
  {
    id:        'rating',
    feature:   'review_scores_rating',
    threshold: 4.8,
    pass:      v => v >= 4.8,
    format:    v => v.toFixed(1),
    label:     'Overall Rating',
  },
  {
    id:        'response',
    feature:   'host_response_rate',
    threshold: 90,
    pass:      v => v >= 90,
    format:    v => v.toFixed(0) + '%',
    label:     'Response Rate',
  },
];

function updateHardRules(sliderVals) {
  const bannerEl = document.getElementById('hard-rules-banner');
  let failedLabels = [];

  HARD_RULES.forEach(rule => {
    const val = sliderVals[rule.feature];
    const passing = rule.pass(val);
    const rowEl  = document.getElementById(`rule-${rule.id}`);
    const iconEl = document.getElementById(`rule-icon-${rule.id}`);
    const valEl  = document.getElementById(`rule-val-${rule.id}`);

    if (!rowEl || !iconEl || !valEl) return;

    rowEl.classList.toggle('rule-pass', passing);
    rowEl.classList.toggle('rule-fail', !passing);
    iconEl.textContent  = passing ? '✅' : '❌';
    iconEl.className    = `rule-icon ${passing ? 'pass' : 'fail'}`;
    valEl.textContent   = rule.format(val);
    valEl.className     = `rule-current ${passing ? 'pass' : 'fail'}`;

    if (!passing) failedLabels.push(rule.label);
  });

  if (!bannerEl) return;

  if (failedLabels.length === 0) {
    bannerEl.hidden    = false;
    bannerEl.className = 'hard-rules-banner pass';
    bannerEl.textContent = '✅ Eligible for Airbnb\'s formal Superhost review.';
  } else {
    bannerEl.hidden    = false;
    bannerEl.className = 'hard-rules-banner warn';
    const names = failedLabels.join(' and ');
    bannerEl.textContent =
      `⚠ Mathematically your market behavior aligns with Superhosts, ` +
      `but Airbnb's hard ${names} threshold${failedLabels.length > 1 ? 's' : ''} ` +
      `would currently disqualify you.`;
  }
}

// Toggle open/close for checklist
function initHardRulesToggle() {
  const btn  = document.getElementById('hard-rules-toggle');
  const body = document.getElementById('hard-rules-body');
  const icon = document.getElementById('hard-rules-toggle-icon');
  if (!btn || !body || !icon) return;
  btn.addEventListener('click', () => {
    const expanded = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!expanded));
    body.hidden = expanded;
    icon.classList.toggle('open', !expanded);
  });
}

// ─────────────────────────────────────────────────────────────────────
// Trigger prediction (debounced)
// ─────────────────────────────────────────────────────────────────────
function triggerPredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(async () => {
    const vals = getSliderValues();
    try {
      const data = await fetchPrediction(vals);
      updateGauge(data.probability);
      updateRadar(vals);
      updateHardRules(vals);
      renderRecommendations(data.recommendations);
    } catch(e) {
      console.error('Predict error:', e);
    }
  }, 150);
}

function triggerSimulate() {
  clearTimeout(scaleTimer);
  scaleTimer = setTimeout(async () => {
    const vals = getSliderValues();
    try {
      const data = await fetchSimulation(vals);
      updateScaleChart(data.curve, data.sweet_spot, vals.host_listings_count);
    } catch(e) {
      console.error('Simulate error:', e);
    }
  }, 400);
}

// ─────────────────────────────────────────────────────────────────────
// Neighbourhood Dropdown — populate from fetched stats
// ─────────────────────────────────────────────────────────────────────
function populateNeighbourhoodDropdown(stats) {
  const select = document.getElementById('neighbourhood-select');
  if (!select) return;
  const counties = Object.keys(stats).sort();
  counties.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name + ' County';
    select.appendChild(opt);
  });
}

// ─────────────────────────────────────────────────────────────────────
// Agent 3 — Smart Task Ticketing
// ─────────────────────────────────────────────────────────────────────
const CAT_ICONS = {
  Maintenance:   '🔧',
  Housekeeping:  '🧹',
  Amenities:     '🛋️',
  Communication: '💬',
};

function agentProbColor(prob) {
  if (prob >= 0.6) return '#2d8f5e';
  if (prob >= 0.4) return '#d97706';
  return '#dc2626';
}

function renderAtRiskCard(listing) {
  const pct = Math.round(listing.probability * 100);
  const col = agentProbColor(listing.probability);
  const safeName = listing.listing_name.replace(/'/g, "\\'").replace(/"/g, '&quot;');
  return `
    <div class="at-risk-card animate-fade-in" id="atrisk-${listing.listing_id}">
      <div class="at-risk-card-header">
        <div class="at-risk-name">${listing.listing_name}</div>
        <div class="at-risk-rating">⭐ ${listing.rating}</div>
      </div>
      <div class="at-risk-meta">
        <span>📍 ${listing.county} County</span>
        <span>💬 ${listing.review_count} reviews</span>
        <span>🏠 ${pct}% SH prob</span>
      </div>
      <div class="at-risk-prob-bar">
        <div class="at-risk-prob-fill" style="width:${pct}%;background:${col};"></div>
      </div>
      <button class="at-risk-generate-btn"
              id="btn-${listing.listing_id}"
              onclick="generateTickets(${listing.listing_id}, '${safeName}', '${listing.county}', ${listing.rating}, ${listing.review_count}, ${pct})">
        ✦ Generate Task Tickets
      </button>
    </div>`;
}

function renderTicketCard(ticket, i) {
  const catClass = 'cat-' + ticket.category.toLowerCase();
  const priClass = 'priority-' + ticket.priority.toLowerCase();
  const icon     = CAT_ICONS[ticket.category] || '📋';
  return `
    <div class="ticket-card animate-fade-in" style="animation-delay:${i*0.08}s">
      <div class="ticket-card-top">
        <span class="ticket-category ${catClass}">${icon} ${ticket.category}</span>
        <span class="ticket-priority ${priClass}">${ticket.priority}</span>
      </div>
      <div class="ticket-root-cause">${ticket.root_cause}</div>
      <div class="ticket-action-label">Recommended Action</div>
      <div class="ticket-action">${ticket.recommended_action}</div>
    </div>`;
}

async function generateTickets(listingId, name, county, rating, reviews, prob) {
  document.querySelectorAll('.at-risk-card').forEach(c => c.classList.remove('selected'));
  const card = document.getElementById('atrisk-' + listingId);
  const btn  = document.getElementById('btn-'    + listingId);
  if (card) card.classList.add('selected');
  if (btn)  { btn.disabled = true; btn.innerHTML = '<div class="btn-spinner"></div> Generating\u2026'; }

  const resultsEl = document.getElementById('ticket-results');
  const nameEl    = document.getElementById('ticket-listing-name');
  const metaEl    = document.getElementById('ticket-listing-meta');
  const gridEl    = document.getElementById('tickets-grid');

  nameEl.textContent = name;
  metaEl.textContent = county + ' County \u00b7 \u2b50 ' + rating + '/5.0 \u00b7 ' + reviews + ' reviews \u00b7 ' + prob + '% Superhost probability';
  gridEl.innerHTML   = '<div class="at-risk-empty"><div class="pulse">Analysing reviews with Groq LLM\u2026</div></div>';
  resultsEl.hidden   = false;
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const res  = await fetch('/agent/tickets/' + listingId, { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (!data.tickets || data.tickets.length === 0) {
      gridEl.innerHTML = '<div class="at-risk-empty">No specific issues identified in the latest reviews.</div>';
    } else {
      gridEl.innerHTML = data.tickets.map((t, i) => renderTicketCard(t, i)).join('');
    }
  } catch(e) {
    gridEl.innerHTML = '<div class="at-risk-empty" style="color:var(--rose);">Error generating tickets: ' + e.message + '</div>';
    console.error('Agent ticket error:', e);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '\u2746 Generate Task Tickets'; }
  }
}

function closeTickets() {
  document.getElementById('ticket-results').hidden = true;
  document.querySelectorAll('.at-risk-card').forEach(c => c.classList.remove('selected'));
}

async function loadAgentSection() {
  const banner = document.getElementById('agent-status-banner');
  const grid   = document.getElementById('at-risk-grid');
  if (!banner || !grid) return;

  let attempts = 0;
  const MAX_ATTEMPTS = 40;

  const poll = async () => {
    try {
      const res  = await fetch('/agent/at-risk');
      const data = await res.json();
      if (data.status === 'ready' && data.listings && data.listings.length > 0) {
        grid.innerHTML = data.listings.map(renderAtRiskCard).join('');
        banner.className = 'agent-status-banner agent-ready';
        banner.innerHTML = '\u2705 Agent ready \u2014 ' + data.listings.length + ' listings identified where review score is the top SHAP negative driver. Click any listing to generate task tickets.';
        return;
      }
      attempts++;
      if (attempts < MAX_ATTEMPTS) setTimeout(poll, 5000);
      else {
        banner.innerHTML = '\u26a0 Agent data took too long to load. Please refresh.';
      }
    } catch(e) {
      attempts++;
      if (attempts < MAX_ATTEMPTS) setTimeout(poll, 5000);
    }
  };
  poll();
}

// ─────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────
async function boot() {
  // Initialise charts
  initGauge();
  initRadar();
  initScaleChart();

  // Load model metadata
  try {
    const mRes = await fetch('/model-info');
    metadata = await mRes.json();

    // Update hero stats — show best model name and its AUC
    const bestName = metadata.best_model_name
      || Object.entries(metadata.model_performance || {})
           .reduce((best, [n, m]) => m.roc_auc > (best[1]?.roc_auc || 0) ? [n, m] : best, ['', null])[0];
    if (bestName && metadata.model_performance?.[bestName]) {
      document.getElementById('stat-auc').textContent =
        metadata.model_performance[bestName].roc_auc.toFixed(3);
      const modelNameEl = document.getElementById('stat-model-name');
      if (modelNameEl) modelNameEl.textContent = bestName;
    }
    if (metadata.superhost_rate) {
      document.getElementById('stat-sh-rate').textContent =
        (metadata.superhost_rate * 100).toFixed(0) + '%';
    }
    if (metadata.total_listings) {
      document.getElementById('stat-listings').textContent =
        metadata.total_listings.toLocaleString();
    }

    // Render feature importance & model table
    if (metadata.feature_importance) renderImportanceChart(metadata.feature_importance);
    if (metadata.model_performance)  renderPerfTable(metadata.model_performance);

    // Pre-populate scale chart from cached metadata
    if (metadata.scale_curve) {
      const peak = metadata.scale_curve.reduce((a,b) => a.probability>b.probability?a:b);
      updateScaleChart(metadata.scale_curve, peak.listings, 3);
    }
  } catch(e) {
    console.error('Could not load model info:', e);
  }

  // Load neighbourhood stats + init map
  try {
    const nRes = await fetch('/neighbourhood-stats');
    if (!nRes.ok) throw new Error('neighbourhood-stats: ' + nRes.status);
    neighStats = await nRes.json();          // store in module state
    populateNeighbourhoodDropdown(neighStats);
    await initMap(neighStats);
  } catch(e) {
    console.warn('Map data not available:', e);
    // Still attempt to init map with empty stats so tiles load
    try {
      await initMap({});
    } catch(e2) {
      const mapEl = document.getElementById('map');
      if (mapEl) {
        mapEl.innerHTML = '<div class="map-status-msg"><span>Map data unavailable. Please ensure the server is running.</span></div>';
      }
    }
  }

  // Attach slider listeners
  SLIDER_IDS.forEach(id => {
    const input = document.getElementById(id);
    if (!input) return;
    input.addEventListener('input', () => {
      updateSliderDisplay(id, parseFloat(input.value));
      triggerPredict();
      triggerSimulate();
    });
    // Initial display
    updateSliderDisplay(id, parseFloat(input.value));
  });

  // Neighbourhood dropdown — update radar benchmark on change
  const neighSelect = document.getElementById('neighbourhood-select');
  if (neighSelect) {
    neighSelect.addEventListener('change', () => {
      const vals = getSliderValues();
      updateRadar(vals);
    });
  }

  // Init hard rules toggle and run initial evaluation
  initHardRulesToggle();

  // Initial prediction
  triggerPredict();
  triggerSimulate();

  // Start Agent 3 background polling
  loadAgentSection();

  // Hide loading overlay
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('hidden');
  setTimeout(() => { overlay.style.display = 'none'; }, 600);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
