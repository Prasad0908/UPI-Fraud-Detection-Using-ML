/* ═══════════════════════════════════════════════════════════
   UPI FraudGuard — Dashboard Interactivity
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
    loadStats();
    loadTransactions();
    setupPredictionForm();
    setupLookupForm();
    setupNavScroll();
    setupSearchToggle();
});

// ── Animated Counter ─────────────────────────────────────────
function animateCounter(el, target, suffix = "") {
    const duration = 1500;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * eased);

        if (target >= 1000) {
            el.textContent = current.toLocaleString("en-IN") + suffix;
        } else {
            el.textContent = current + suffix;
        }

        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function animateCounterFloat(el, target, suffix = "%") {
    const duration = 1500;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = (target * eased).toFixed(2);
        el.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ── Load Stats & Model Cards ──────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        const ds = data.dataset;

        const totalEl = document.querySelector("#stat-total .stat-value");
        const fraudEl = document.querySelector("#stat-fraud .stat-value");
        const legitEl = document.querySelector("#stat-legit .stat-value");
        const accEl = document.querySelector("#stat-accuracy .stat-value");

        animateCounter(totalEl, ds.total_transactions);
        animateCounter(fraudEl, ds.fraud_count);
        animateCounter(legitEl, ds.legit_count);

        const bestName = data.best_model;
        const bestAuc = data.models[bestName].auc_roc * 100;
        animateCounterFloat(accEl, bestAuc, "%");

        buildModelCards(data.models, bestName);

    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

function buildModelCards(models, bestName) {
    const container = document.getElementById("model-cards");
    container.innerHTML = "";

    const metricLabels = {
        accuracy: "Accuracy",
        precision: "Precision",
        recall: "Recall",
        f1_score: "F1 Score",
        auc_roc: "AUC-ROC"
    };

    Object.entries(models).forEach(([name, metrics], index) => {
        const isBest = name === bestName;
        const card = document.createElement("div");
        card.className = `model-card${isBest ? " best" : ""}`;
        card.style.animationDelay = `${index * 0.15}s`;

        let metricsHTML = "";
        Object.entries(metricLabels).forEach(([key, label]) => {
            const val = metrics[key];
            const pct = (val * 100).toFixed(1);
            metricsHTML += `
                <div class="metric-row">
                    <span class="metric-label">${label}</span>
                    <span class="metric-value">${pct}%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: 0%;" data-width="${pct}%"></div>
                </div>
            `;
        });

        card.innerHTML = `<div class="model-name">${name}</div>${metricsHTML}`;
        container.appendChild(card);

        setTimeout(() => {
            card.querySelectorAll(".metric-fill").forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 300 + index * 200);
    });
}

// ── Load Transactions ────────────────────────────────────────
async function loadTransactions() {
    try {
        const res = await fetch("/api/recent");
        const data = await res.json();
        const tbody = document.getElementById("transactions-body");
        tbody.innerHTML = "";

        data.forEach(txn => {
            const isFraud = txn.is_fraud === 1;
            const row = document.createElement("tr");
            if (isFraud) row.className = "fraud-row";

            row.innerHTML = `
                <td>${txn.transaction_id}</td>
                <td>₹${parseFloat(txn.amount).toLocaleString("en-IN", {minimumFractionDigits: 2})}</td>
                <td>${txn.transaction_type}</td>
                <td>${txn.sender_bank}</td>
                <td>${txn.device_type}</td>
                <td>${txn.sender_city}</td>
                <td><span class="status-badge ${isFraud ? 'fraud' : 'legit'}">${isFraud ? '🚨 Fraud' : '✅ Legit'}</span></td>
            `;
            tbody.appendChild(row);
        });
    } catch (err) {
        console.error("Failed to load transactions:", err);
    }
}

// ═══════════════════════════════════════════════════════════════
// ── Tab Switching ────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

function switchTab(tab) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    document.getElementById(`tab-${tab}`).classList.add("active");
    document.getElementById(`content-${tab}`).classList.add("active");
}

// ═══════════════════════════════════════════════════════════════
// ── Search Type Toggle ───────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

function setupSearchToggle() {
    const radios = document.querySelectorAll('input[name="search_type"]');
    const queryInput = document.getElementById("lookup-query");

    radios.forEach(radio => {
        radio.addEventListener("change", () => {
            document.querySelectorAll(".toggle-option").forEach(t => t.classList.remove("active"));
            radio.closest(".toggle-option").classList.add("active");

            if (radio.value === "transaction_id") {
                queryInput.placeholder = "Enter Transaction ID (e.g. UPI00000001)";
            } else {
                queryInput.placeholder = "Enter UPI ID (e.g. rahul123@oksbi)";
            }
        });
    });
}

// ── Fill Example ─────────────────────────────────────────────
function fillExample(value, type) {
    document.getElementById("lookup-query").value = value;
    const radio = document.querySelector(`input[name="search_type"][value="${type}"]`);
    if (radio) {
        radio.checked = true;
        radio.dispatchEvent(new Event("change"));
    }
}

// ═══════════════════════════════════════════════════════════════
// ── Lookup Form (Verify Payment) ─────────────────────────────
// ═══════════════════════════════════════════════════════════════

function setupLookupForm() {
    const form = document.getElementById("lookup-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const btn = document.getElementById("lookup-btn");
        btn.querySelector(".btn-text").style.display = "none";
        btn.querySelector(".btn-loader").style.display = "inline";

        const query = document.getElementById("lookup-query").value.trim();
        const searchType = document.querySelector('input[name="search_type"]:checked').value;

        try {
            const res = await fetch("/api/lookup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, search_type: searchType }),
            });
            const data = await res.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            showLookupResult(data);

        } catch (err) {
            console.error("Lookup failed:", err);
            alert("Failed to verify payment. Please try again.");
        } finally {
            btn.querySelector(".btn-text").style.display = "inline";
            btn.querySelector(".btn-loader").style.display = "none";
        }
    });
}

function showLookupResult(data) {
    const resultPanel = document.getElementById("lookup-result");
    const notFound = document.getElementById("lookup-not-found");
    const found = document.getElementById("lookup-found");

    resultPanel.style.display = "block";

    if (!data.found) {
        // Not found — likely fake payment
        notFound.style.display = "block";
        found.style.display = "none";
        document.getElementById("not-found-msg").textContent = data.message;
        resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
    }

    // Found
    notFound.style.display = "none";
    found.style.display = "block";

    const txn = data.transactions[0];
    const isFraud = txn.is_fraud === 1;
    const mlProb = txn.ml_fraud_probability;

    // ── Verdict Banner ──
    const banner = document.getElementById("verdict-banner");
    const icon = document.getElementById("verdict-icon");
    const title = document.getElementById("verdict-title");
    const subtitle = document.getElementById("verdict-subtitle");
    const prob = document.getElementById("verdict-prob");

    banner.className = "verdict-banner";
    if (isFraud) {
        banner.classList.add("verdict-fraud");
        icon.textContent = "🚨";
        title.textContent = "FRAUDULENT TRANSACTION";
        subtitle.textContent = "This transaction has been flagged as fraudulent. Do NOT deliver goods!";
        prob.textContent = mlProb !== null ? mlProb.toFixed(1) + "%" : "—";
        prob.className = "verdict-prob fraud";
    } else if (mlProb !== null && mlProb > 50) {
        banner.classList.add("verdict-warning");
        icon.textContent = "⚠️";
        title.textContent = "SUSPICIOUS TRANSACTION";
        subtitle.textContent = "ML model flags this transaction as potentially suspicious. Verify with your bank.";
        prob.textContent = mlProb.toFixed(1) + "%";
        prob.className = "verdict-prob warning";
    } else {
        banner.classList.add("verdict-safe");
        icon.textContent = "✅";
        title.textContent = "PAYMENT VERIFIED";
        subtitle.textContent = "This transaction appears legitimate. Safe to proceed.";
        prob.textContent = mlProb !== null ? mlProb.toFixed(1) + "%" : "—";
        prob.className = "verdict-prob safe";
    }

    // ── Transaction Details Grid ──
    const grid = document.getElementById("txn-details-grid");
    grid.innerHTML = `
        <div class="detail-item">
            <span class="detail-label">Transaction ID</span>
            <span class="detail-value">${txn.transaction_id}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Date & Time</span>
            <span class="detail-value">${new Date(txn.timestamp).toLocaleString("en-IN")}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Amount</span>
            <span class="detail-value highlight">₹${txn.amount.toLocaleString("en-IN", {minimumFractionDigits: 2})}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Transaction Type</span>
            <span class="detail-value">${txn.transaction_type}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Sender UPI ID</span>
            <span class="detail-value mono">${txn.sender_upi_id}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Receiver UPI ID</span>
            <span class="detail-value mono">${txn.receiver_upi_id}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Sender Bank</span>
            <span class="detail-value">${txn.sender_bank}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Receiver Bank</span>
            <span class="detail-value">${txn.receiver_bank}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Device</span>
            <span class="detail-value">${txn.device_type}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">UPI App</span>
            <span class="detail-value">${txn.upi_app}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Sender City</span>
            <span class="detail-value">${txn.sender_city}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Receiver City</span>
            <span class="detail-value">${txn.receiver_city}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Sender Balance Before</span>
            <span class="detail-value">₹${txn.sender_balance_before.toLocaleString("en-IN", {minimumFractionDigits: 2})}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Sender Balance After</span>
            <span class="detail-value">₹${txn.sender_balance_after.toLocaleString("en-IN", {minimumFractionDigits: 2})}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">ML Fraud Probability</span>
            <span class="detail-value ${mlProb > 50 ? 'text-danger' : 'text-safe'}">${mlProb !== null ? mlProb.toFixed(2) + '%' : 'N/A'}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Dataset Label</span>
            <span class="detail-value"><span class="status-badge ${isFraud ? 'fraud' : 'legit'}">${isFraud ? '🚨 Fraud' : '✅ Legitimate'}</span></span>
        </div>
    `;

    // ── UPI ID Summary (for UPI searches with multiple results) ──
    const upiSummary = document.getElementById("upi-summary");
    const txnList = document.getElementById("lookup-txn-list");

    if (data.search_type === "upi_id" && data.summary) {
        const s = data.summary;
        upiSummary.style.display = "block";
        document.getElementById("upi-summary-stats").innerHTML = `
            <div class="summary-stat">
                <div class="summary-stat-value">${s.total_transactions}</div>
                <div class="summary-stat-label">Total Transactions</div>
            </div>
            <div class="summary-stat ${s.fraud_count > 0 ? 'danger' : ''}">
                <div class="summary-stat-value">${s.fraud_count}</div>
                <div class="summary-stat-label">Fraudulent</div>
            </div>
            <div class="summary-stat safe">
                <div class="summary-stat-value">${s.legit_count}</div>
                <div class="summary-stat-label">Legitimate</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">₹${s.total_amount.toLocaleString("en-IN")}</div>
                <div class="summary-stat-label">Total Amount</div>
            </div>
            <div class="summary-stat ${s.fraud_rate > 0 ? 'danger' : 'safe'}">
                <div class="summary-stat-value">${s.fraud_rate}%</div>
                <div class="summary-stat-label">Fraud Rate</div>
            </div>
            <div class="summary-stat ${s.is_suspicious ? 'danger' : 'safe'}">
                <div class="summary-stat-value">${s.is_suspicious ? '⚠️ YES' : '✅ NO'}</div>
                <div class="summary-stat-label">Suspicious?</div>
            </div>
        `;

        // Show transaction list table
        if (data.transactions.length > 1) {
            txnList.style.display = "block";
            const tbody = document.getElementById("lookup-table-body");
            tbody.innerHTML = "";

            data.transactions.forEach(t => {
                const f = t.is_fraud === 1;
                const row = document.createElement("tr");
                if (f) row.className = "fraud-row";

                const mlScore = t.ml_fraud_probability !== null ? t.ml_fraud_probability.toFixed(1) + "%" : "—";
                const mlClass = t.ml_fraud_probability > 50 ? "text-danger" : "text-safe";

                row.innerHTML = `
                    <td>${t.transaction_id}</td>
                    <td>${new Date(t.timestamp).toLocaleDateString("en-IN")}</td>
                    <td>₹${t.amount.toLocaleString("en-IN", {minimumFractionDigits: 2})}</td>
                    <td>${t.transaction_type}</td>
                    <td class="mono-sm">${t.sender_upi_id}</td>
                    <td class="mono-sm">${t.receiver_upi_id}</td>
                    <td class="${mlClass}">${mlScore}</td>
                    <td><span class="status-badge ${f ? 'fraud' : 'legit'}">${f ? '🚨 Fraud' : '✅ Legit'}</span></td>
                `;
                tbody.appendChild(row);
            });
        } else {
            txnList.style.display = "none";
        }
    } else {
        upiSummary.style.display = "none";
        txnList.style.display = "none";
    }

    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ═══════════════════════════════════════════════════════════════
// ── Manual Prediction Form ───────────────────────────────────
// ═══════════════════════════════════════════════════════════════

function setupPredictionForm() {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const btn = document.getElementById("predict-btn");
        btn.querySelector(".btn-text").style.display = "none";
        btn.querySelector(".btn-loader").style.display = "inline";

        const payload = {
            amount: document.getElementById("amount").value,
            sender_balance: document.getElementById("sender_balance").value,
            receiver_balance: document.getElementById("receiver_balance").value,
            hour: document.getElementById("hour").value,
            transaction_type: document.getElementById("transaction_type").value,
            device_type: document.getElementById("device_type").value,
            sender_bank: document.getElementById("sender_bank").value,
            receiver_bank: document.getElementById("receiver_bank").value,
            sender_city: document.getElementById("sender_city").value,
            upi_app: document.getElementById("upi_app").value,
            receiver_city: "Delhi",
            day_of_week: new Date().getDay(),
            month: new Date().getMonth() + 1,
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const result = await res.json();
            showPredictionResult(result);
        } catch (err) {
            console.error("Prediction failed:", err);
        } finally {
            btn.querySelector(".btn-text").style.display = "inline";
            btn.querySelector(".btn-loader").style.display = "none";
        }
    });
}

function showPredictionResult(result) {
    const panel = document.getElementById("predict-result");
    panel.style.display = "block";

    const prob = result.fraud_probability;
    const gaugeValue = document.getElementById("gauge-value");
    const gaugeFill = document.getElementById("gauge-fill");
    const badge = document.getElementById("result-badge");
    const risk = document.getElementById("result-risk");

    const maxOffset = 251.2;
    const targetOffset = maxOffset - (maxOffset * (prob / 100));

    animateCounterFloat(gaugeValue, prob, "%");

    gaugeFill.style.transition = "stroke-dashoffset 1.5s ease";
    gaugeFill.setAttribute("stroke-dashoffset", targetOffset);

    if (prob < 30) {
        gaugeValue.style.color = "#10b981";
    } else if (prob < 60) {
        gaugeValue.style.color = "#f59e0b";
    } else {
        gaugeValue.style.color = "#f43f5e";
    }

    badge.textContent = result.prediction;
    badge.className = "result-badge";
    if (result.is_fraud) {
        badge.classList.add("danger");
    } else if (prob > 30) {
        badge.classList.add("warning");
    } else {
        badge.classList.add("safe");
    }

    risk.textContent = `Risk Level: ${result.risk_level}`;
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ── Navbar Active Link on Scroll ─────────────────────────────
function setupNavScroll() {
    const sections = document.querySelectorAll("section, header");
    const navLinks = document.querySelectorAll(".nav-link");

    window.addEventListener("scroll", () => {
        let current = "";
        sections.forEach(section => {
            const top = section.offsetTop - 150;
            if (window.scrollY >= top) {
                current = section.getAttribute("id");
            }
        });

        navLinks.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === "#" + current) {
                link.classList.add("active");
            }
        });
    });
}
