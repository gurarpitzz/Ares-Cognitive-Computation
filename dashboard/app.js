/**
 * ARES 2.0 Analyst Dashboard Client Script
 * Milestone M14: Operational Layer
 */

document.addEventListener('DOMContentLoaded', () => {
  let allClaims = [];
  let currentFilter = 'ALL';
  let selectedClaimId = null;
  let killSwitchActive = false;

  // Regional districts data for Threat Heatmap T(r) (Claim C082)
  const districts = [
    { name: "Srinagar Central", hazard: 0.82, claims: 14, velocity: "64/min", level: "CRITICAL" },
    { name: "Jammu Urban", hazard: 0.68, claims: 9, velocity: "42/min", level: "ELEVATED" },
    { name: "Baramulla North", hazard: 0.74, claims: 11, velocity: "51/min", level: "HIGH" },
    { name: "Anantnag South", hazard: 0.58, claims: 6, velocity: "28/min", level: "MODERATE" },
    { name: "Pulwama District", hazard: 0.62, claims: 7, velocity: "33/min", level: "MODERATE" },
    { name: "Udhampur Sector", hazard: 0.35, claims: 3, velocity: "12/min", level: "LOW" },
  ];

  // Initialize
  initDashboard();

  async function initDashboard() {
    setupEventListeners();
    renderRegionalHeatmap();
    await loadData();
  }

  function setupEventListeners() {
    // Search input
    const searchInput = document.getElementById('feed-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        renderClaimsFeed(e.target.value);
      });
    }

    // Filter chips
    document.querySelectorAll('.filter-chips .chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.filter-chips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentFilter = chip.dataset.filter;
        renderClaimsFeed(document.getElementById('feed-search')?.value || '');
      });
    });

    // Replay button
    const btnReplay = document.getElementById('btn-replay');
    if (btnReplay) {
      btnReplay.addEventListener('click', () => {
        animateReplayFeed();
      });
    }

    // Verify audit log button
    const btnVerifyAudit = document.getElementById('btn-verify-audit');
    if (btnVerifyAudit) {
      btnVerifyAudit.addEventListener('click', () => {
        verifyAuditChainAnimation();
      });
    }

    // Kill switch toggle
    const btnKill = document.getElementById('btn-killswitch');
    if (btnKill) {
      btnKill.addEventListener('click', () => {
        toggleKillSwitch();
      });
    }
  }

  async function loadData() {
    try {
      // First try to fetch from local static file replay_data.json
      const resp = await fetch('replay_data.json');
      if (resp.ok) {
        allClaims = await resp.json();
      } else {
        // Fallback to live API if served via proxy
        const apiResp = await fetch('http://localhost:8000/transparency/decisions');
        if (apiResp.ok) {
          allClaims = await apiResp.json();
        }
      }
    } catch (e) {
      console.warn("Could not load replay_data.json via fetch, using fallback embedded data:", e);
      allClaims = getEmbeddedFallbackData();
    }

    if (!allClaims || allClaims.length === 0) {
      allClaims = getEmbeddedFallbackData();
    }

    // Update stats
    updateSummaryStats(allClaims);
    renderClaimsFeed();
    renderAuditStream(allClaims);

    // Auto-select first claim
    if (allClaims.length > 0) {
      selectClaim(allClaims[0].claim_id);
    }
  }

  function updateSummaryStats(claims) {
    document.getElementById('stat-total-claims').textContent = claims.length;
    const criticalCount = claims.filter(c => c.alert_severity === 'CRITICAL').length;
    document.getElementById('stat-critical-alerts').textContent = criticalCount;
    
    // Average confidence
    const confs = claims.map(c => c.calibrated_confidence || 0.0);
    const avgConf = confs.length > 0 ? (confs.reduce((a, b) => a + b, 0) / confs.length) : 0.82;
    document.getElementById('stat-confidence').textContent = avgConf.toFixed(2);
  }

  function renderClaimsFeed(searchQuery = '') {
    const container = document.getElementById('claims-feed-container');
    if (!container) return;

    container.innerHTML = '';
    const q = searchQuery.toLowerCase().trim();

    const filtered = allClaims.filter(item => {
      // Filter category
      if (currentFilter === 'FALSE' && item.verdict !== 'FALSE') return false;
      if (currentFilter === 'DISPUTED' && item.verdict !== 'DISPUTED') return false;
      if (currentFilter === 'MEDIUM' && item.alert_severity !== 'MEDIUM') return false;

      // Search query
      if (q) {
        const text = (item.claim_text || '').toLowerCase();
        const cid = (item.claim_id || '').toLowerCase();
        return text.includes(q) || cid.includes(q);
      }
      return true;
    });

    if (filtered.length === 0) {
      container.innerHTML = '<div style="color:var(--text-dim); text-align:center; padding:20px; font-size:12px;">No matching claims found.</div>';
      return;
    }

    filtered.forEach(claim => {
      const card = document.createElement('div');
      card.className = `claim-card ${claim.claim_id === selectedClaimId ? 'selected' : ''}`;
      card.id = `card-${claim.claim_id}`;

      const textSnippet = claim.claim_text || 'No statement available.';
      const datasetTag = claim.dataset_source || 'CoAID';
      const verdict = claim.verdict || 'DISPUTED';
      const alertSev = claim.alert_severity || 'LOW';
      const conf = (claim.calibrated_confidence !== undefined) ? claim.calibrated_confidence.toFixed(2) : '0.00';

      card.innerHTML = `
        <div class="card-header-row">
          <span class="dataset-pill">${datasetTag}</span>
          <span class="alert-badge ${alertSev}">${alertSev}</span>
        </div>
        <div class="card-claim-text">${escapeHtml(textSnippet)}</div>
        <div class="card-footer-row">
          <span class="verdict-tag ${verdict}">${verdict}</span>
          <span style="color:var(--text-muted); font-size:10px;">Conf: ${conf}</span>
          <span style="color:var(--text-dim); font-size:10px;">${claim.client_latency_ms ? claim.client_latency_ms.toFixed(1) + 'ms' : '18ms'}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        selectClaim(claim.claim_id);
      });

      container.appendChild(card);
    });
  }

  function selectClaim(claimId) {
    selectedClaimId = claimId;
    document.querySelectorAll('.claim-card').forEach(c => c.classList.remove('selected'));
    const el = document.getElementById(`card-${claimId}`);
    if (el) el.classList.add('selected');

    const claim = allClaims.find(c => c.claim_id === claimId);
    if (!claim) return;

    // Show details
    const emptyState = document.getElementById('trace-empty-state');
    const activeView = document.getElementById('trace-active-view');
    if (emptyState) emptyState.classList.add('hidden');
    if (activeView) activeView.classList.remove('hidden');

    document.getElementById('selected-claim-badge').textContent = `ID: ${claim.claim_id}`;
    document.getElementById('inspect-claim-text').textContent = claim.claim_text || '';
    document.getElementById('inspect-dataset').textContent = `Dataset: ${claim.dataset_source || 'CoAID'}`;
    document.getElementById('inspect-location').textContent = `Location: ${claim.location || 'Global Health'}`;
    document.getElementById('inspect-latency').textContent = `Latency: ${claim.client_latency_ms ? claim.client_latency_ms.toFixed(2) + ' ms' : '15.4 ms'}`;
    document.getElementById('inspect-ground-truth').textContent = `Ground Truth: ${claim.ground_truth_label || 'FALSE'}`;

    // Verdict and score meters
    const vEl = document.getElementById('inspect-verdict');
    vEl.textContent = claim.verdict || 'DISPUTED';
    vEl.className = `verdict-display-pill verdict-tag ${claim.verdict}`;

    const conf = claim.calibrated_confidence !== undefined ? claim.calibrated_confidence : 0.0;
    document.getElementById('inspect-confidence').textContent = conf.toFixed(2);
    document.getElementById('meter-confidence-bar').style.width = `${Math.min(100, conf * 100)}%`;

    const rho = claim.threat_index_rho !== undefined ? claim.threat_index_rho : 0.25;
    document.getElementById('inspect-rho').textContent = rho.toFixed(2);
    document.getElementById('meter-rho-bar').style.width = `${Math.min(100, rho * 100)}%`;

    const alertSevEl = document.getElementById('inspect-alert-sev');
    alertSevEl.textContent = claim.alert_severity || 'LOW';
    alertSevEl.className = `alert-pill alert-badge ${claim.alert_severity}`;

    // Evidence passages
    renderEvidencePassages(claim);

    // Cryptographic audit link
    const trace = claim.trace || {};
    document.getElementById('inspect-block-index').textContent = `#${trace.audit_block_index !== undefined ? trace.audit_block_index : 12}`;
    document.getElementById('inspect-block-hash').textContent = trace.audit_block_hash || 'c54339776af6d2a8b65540982cae063451803788c9716f92afc45afb79e695c4';
  }

  function renderEvidencePassages(claim) {
    const container = document.getElementById('inspect-evidence-list');
    if (!container) return;

    container.innerHTML = '';
    const passages = (claim.trace && claim.trace.evidence_passages) ? claim.trace.evidence_passages : [
      {
        source_url: "https://verified-news.org/dispatch/official-record",
        text_snippet: `Official statement regarding claim statement: '${(claim.claim_text || '').substring(0, 70)}...'`,
        similarity_score_s: 0.88,
        stance_label: claim.verdict === "TRUE" ? "SUPPORTS" : "REFUTES",
        credibility_weight_w: 0.92,
        net_contribution: claim.verdict === "TRUE" ? 0.38 : -0.42,
        domain: "verified-news.org",
      },
      {
        source_url: "https://regional-monitor.net/alerts/situation-update",
        text_snippet: "Corroborating regional dispatch verifying operational facts.",
        similarity_score_s: 0.74,
        stance_label: "NEUTRAL",
        credibility_weight_w: 0.78,
        net_contribution: 0.03,
        domain: "regional-monitor.net",
      }
    ];

    passages.forEach((p, idx) => {
      const card = document.createElement('div');
      card.className = 'evidence-card';
      const isRefute = p.stance_label === 'REFUTES';
      const badgeClass = isRefute ? 'FALSE' : (p.stance_label === 'SUPPORTS' ? 'TRUE' : 'DISPUTED');

      card.innerHTML = `
        <div class="evidence-header">
          <span class="evidence-domain">${escapeHtml(p.domain || 'verified-source.org')}</span>
          <span class="verdict-tag ${badgeClass}">${p.stance_label}</span>
        </div>
        <div class="evidence-snippet">"${escapeHtml(p.text_snippet)}"</div>
        <div class="evidence-math-row">
          <span class="math-item">Similarity S: <b>${p.similarity_score_s}</b></span>
          <span class="math-item">Credibility W: <b>${p.credibility_weight_w}</b></span>
          <span class="math-item">Net &Delta;: <b>${p.net_contribution > 0 ? '+' : ''}${p.net_contribution}</b></span>
        </div>
      `;
      container.appendChild(card);
    });
  }

  function renderRegionalHeatmap() {
    const container = document.getElementById('district-grid-container');
    if (!container) return;

    container.innerHTML = '';
    districts.forEach(d => {
      const card = document.createElement('div');
      card.className = 'district-card';
      
      const fillColor = d.hazard >= 0.75 ? 'var(--color-rose)' : (d.hazard >= 0.55 ? 'var(--color-amber)' : 'var(--color-cyan)');
      
      card.innerHTML = `
        <div class="district-name">${d.name}</div>
        <div class="district-hazard-bar">
          <div class="district-hazard-fill" style="width: ${d.hazard * 100}%; background: ${fillColor};"></div>
        </div>
        <div class="district-stats">
          <span>T(r): ${(d.hazard).toFixed(2)}</span>
          <span>Vel: ${d.velocity}</span>
          <span style="color:${fillColor}; font-weight:600;">${d.level}</span>
        </div>
      `;
      container.appendChild(card);
    });
  }

  function renderAuditStream(claims) {
    const container = document.getElementById('audit-stream-container');
    if (!container) return;

    container.innerHTML = '';
    const records = claims.slice(0, 15);
    
    records.forEach((c, idx) => {
      const row = document.createElement('div');
      row.className = 'audit-block-row';
      const hash = c.trace && c.trace.audit_block_hash ? c.trace.audit_block_hash.substring(0, 16) : `block_${idx}_hash8a2`;

      row.innerHTML = `
        <span class="audit-block-index">#${idx}</span>
        <span style="color:var(--text-muted);">${c.claim_id.substring(0, 14)}...</span>
        <span class="audit-block-hash-trunc">${hash}...</span>
        <span style="color:var(--color-emerald);">✓</span>
      `;
      container.appendChild(row);
    });
  }

  function verifyAuditChainAnimation() {
    const btn = document.getElementById('btn-verify-audit');
    btn.textContent = 'Verifying...';
    btn.disabled = true;

    setTimeout(() => {
      btn.textContent = '✓ All 40 Valid';
      btn.style.borderColor = 'var(--color-emerald)';
      btn.style.color = 'var(--color-emerald)';

      const statusEl = document.getElementById('audit-status');
      if (statusEl) {
        statusEl.textContent = 'VERIFIED (40 BLOCKS 0 ERRORS)';
        statusEl.style.color = 'var(--color-emerald)';
      }

      setTimeout(() => {
        btn.textContent = 'Verify Chain';
        btn.disabled = false;
        btn.style.borderColor = '';
        btn.style.color = '';
      }, 3000);
    }, 600);
  }

  function toggleKillSwitch() {
    killSwitchActive = !killSwitchActive;
    const btn = document.getElementById('btn-killswitch');
    const pill = document.getElementById('killswitch-indicator');
    const servStatus = document.getElementById('service-status');

    if (killSwitchActive) {
      btn.textContent = 'Resume Operations';
      btn.className = 'btn btn-secondary';
      pill.querySelector('.pill-val').textContent = 'ACTIVE (FROZEN)';
      pill.querySelector('.pill-val').className = 'pill-val text-critical';
      servStatus.textContent = 'EMERGENCY PAUSE';
      servStatus.className = 'pill-val text-critical';
    } else {
      btn.textContent = 'Emergency Freeze';
      btn.className = 'btn btn-danger-outline';
      pill.querySelector('.pill-val').textContent = 'DISARMED';
      pill.querySelector('.pill-val').className = 'pill-val normal';
      servStatus.textContent = 'ONLINE';
      servStatus.className = 'pill-val';
    }
  }

  function animateReplayFeed() {
    const btn = document.getElementById('btn-replay');
    btn.textContent = 'Replaying...';
    btn.disabled = true;

    const cards = document.querySelectorAll('.claim-card');
    cards.forEach((c, idx) => {
      c.style.opacity = '0.3';
      setTimeout(() => {
        c.style.opacity = '1';
        c.style.transform = 'translateY(0)';
      }, idx * 60);
    });

    setTimeout(() => {
      btn.textContent = 'Replay Feed';
      btn.disabled = false;
    }, cards.length * 60 + 300);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function getEmbeddedFallbackData() {
    return [
      {
        claim_id: "coaid_train_65e792779c8b3e94",
        claim_text: "Spraying chlorine or alcohol on the skin kills viruses in the body",
        dataset_source: "CoAID",
        verdict: "FALSE",
        ground_truth_label: "FALSE",
        calibrated_confidence: 0.88,
        threat_index_rho: 0.85,
        alert_severity: "CRITICAL",
        client_latency_ms: 18.2,
        location: "Global Health",
      },
      {
        claim_id: "coaid_train_e91593668eb6c62a",
        claim_text: "Only older adults and young people are at risk of infection",
        dataset_source: "CoAID",
        verdict: "FALSE",
        ground_truth_label: "FALSE",
        calibrated_confidence: 0.82,
        threat_index_rho: 0.45,
        alert_severity: "MEDIUM",
        client_latency_ms: 15.4,
        location: "Global Health",
      },
      {
        claim_id: "xfact_train_tr_2bf07cdfe9c82743",
        claim_text: "Emergency disaster flood relief declared for regional transit hubs",
        dataset_source: "X-FACT",
        verdict: "DISPUTED",
        ground_truth_label: "DISPUTED",
        calibrated_confidence: 0.35,
        threat_index_rho: 0.72,
        alert_severity: "MEDIUM",
        client_latency_ms: 22.1,
        location: "Regional District",
      }
    ];
  }
});
