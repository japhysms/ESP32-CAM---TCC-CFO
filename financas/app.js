/* Minhas Finanças — app local (sem servidor). Os dados ficam no localStorage do navegador;
 * use Configurações > Backup para exportar/importar um JSON. */
'use strict';

/* =================================================================== utils */
const $ = (s, r = document) => r.querySelector(s);
const pad = n => String(n).padStart(2, '0');
const uid = () => Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-4);
const BRL = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' });
const NUM = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 8 });
const PREFS_KEY = 'minhas-financas.prefs';
const prefs = (() => { try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; } })();
function savePrefs() { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* navegação privada */ } }
const fmt = v => (prefs.hide ? 'R$ •••••' : BRL.format(v || 0));
const fmtNum = v => NUM.format(v || 0);
const pct = v => (isFinite(v) ? (v * 100).toFixed(1).replace('.', ',') : '0,0') + '%';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const norm = Importers.norm;
const round2 = v => Math.round(v * 100) / 100;
const sum = (arr, f = x => x) => arr.reduce((a, x) => a + f(x), 0);

function fmtShort(v) {
  if (prefs.hide) return '•••';
  const a = Math.abs(v);
  if (a >= 1e6) return 'R$ ' + (v / 1e6).toFixed(1).replace('.', ',') + 'mi';
  if (a >= 1e3) return 'R$ ' + (v / 1e3).toFixed(1).replace('.', ',') + 'mil';
  return 'R$ ' + Math.round(v);
}
function todayISO() { const d = new Date(); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`; }
const currentMonth = () => todayISO().slice(0, 7);
function addMonths(ym, n) { let [y, m] = ym.split('-').map(Number); m += n; y += Math.floor((m - 1) / 12); m = ((m - 1) % 12 + 12) % 12 + 1; return `${y}-${pad(m)}`; }
function daysInMonth(ym) { const [y, m] = ym.split('-').map(Number); return new Date(y, m, 0).getDate(); }
const MONTHS = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'];
const monthLabel = ym => { const [y, m] = ym.split('-').map(Number); return `${MONTHS[m - 1]} ${y}`; };
const monthShort = ym => { const [y, m] = ym.split('-').map(Number); return `${MONTHS[m - 1].slice(0, 3)}/${String(y).slice(2)}`; };
const fmtDate = iso => iso ? iso.split('-').reverse().join('/') : '';

/* ================================================================ defaults */
const DEFAULT_TAGS = [
  ['Salário', '#16a34a'], ['Outras receitas', '#65a30d'], ['Investimentos', '#2563eb', 'investimento'], ['Moradia', '#6366f1', 'essencial'],
  ['Alimentação', '#f59e0b', 'essencial'], ['Transporte', '#0ea5e9', 'essencial'], ['Lazer', '#ec4899', 'livre'], ['Saúde', '#10b981', 'essencial'],
  ['Educação', '#8b5cf6', 'essencial'], ['Assinaturas', '#f97316', 'livre'], ['Compras', '#e11d48', 'livre'], ['Impostos e taxas', '#78716c', 'essencial'],
  ['Outros', '#a3a3a3', 'livre'], ['Transferência', '#64748b', '', true],
].map(([name, color, group, exclude]) => ({ name, color, group: group || '', exclude: !!exclude }));
// Grupos da regra 50/30/20 (equilíbrio financeiro)
const GROUPS = { essencial: 'Essencial', livre: 'Não essencial', investimento: 'Investimento' };
const ACC_TYPES = { corrente: 'Conta corrente / poupança', cartao: 'Cartão de crédito', corretora: 'Corretora' };
const NO_TAG = { name: 'Sem Tags', color: '#94a3b8', icon: '🏷️' };
const DEFAULT_ICONS = {
  'Salário': '💼', 'Outras receitas': '💰', 'Investimentos': '📈', 'Moradia': '🏠', 'Alimentação': '🍽️', 'Transporte': '🚗',
  'Lazer': '🎉', 'Saúde': '🩺', 'Educação': '🎓', 'Assinaturas': '📺', 'Compras': '🛍️', 'Impostos e taxas': '🧾',
  'Outros': '📦', 'Transferência': '🔁',
};
DEFAULT_TAGS.forEach(t => { t.icon = DEFAULT_ICONS[t.name] || '🏷️'; });
const ICON_CHOICES = ['🍽️', '🛒', '🍔', '☕', '🍺', '🏠', '💡', '💧', '📶', '🚗', '⛽', '🚌', '🚕', '✈️', '🩺', '💊', '🏋️', '🎓', '📚',
  '🎉', '🎬', '🎮', '🎵', '📺', '🛍️', '👕', '💄', '🎁', '🐾', '👶', '💇', '🔧', '🧹', '❤️', '⛪', '🧾', '🏦', '💳', '💼', '💰', '📈', '🪙', '🔁', '📦', '🏷️'];
function iconBadge(icon, color, cls = '') { return `<span class="ticon ${cls}" style="--c:${color}">${esc(icon || '🏷️')}</span>`; }
function tagIcon(name, cls = '') { const t = tagInfo(name); return iconBadge(t.icon, t.color, cls); }

// [padrões separados por |, tag, sinal: any | in | out]
const DEFAULT_RULES = [
  ['salario|folha de pagamento|proventos salariais', 'Salário', 'in'],
  ['rendimento|resgate|dividendo|juros sobre capital', 'Investimentos', 'in'],
  ['corretora|xp invest|rico invest|btg pactual|nu invest|clear corretora|tesouro direto|aplicacao|cdb|lci|lca', 'Investimentos', 'out'],
  ['pagamento de fatura|pagamento fatura|pagamento recebido|transferencia entre contas|mesma titularidade', 'Transferência'],
  ['aluguel|condominio|enel|light|cemig|copel|sabesp|comgas|energia|agua e esgoto|claro|vivo|tim |internet', 'Moradia', 'out'],
  ['ifood|rappi|restaurante|lanchonete|padaria|mercado|supermercado|carrefour|pao de acucar|assai|atacadao|hortifruti|acougue', 'Alimentação', 'out'],
  ['uber|99app|99 pop|99*|posto|shell|ipiranga|petrobras|estacionamento|sem parar|veloe|metro|bilhete unico', 'Transporte', 'out'],
  ['netflix|spotify|disney|prime video|amazon prime|youtube premium|hbo|max.com|apple.com|google one|icloud|chatgpt|deezer', 'Assinaturas', 'out'],
  ['farmacia|drogasil|droga raia|drogaria|pague menos|hospital|clinica|laboratorio|unimed|amil|sulamerica|dentista', 'Saúde', 'out'],
  ['escola|faculdade|universidade|curso|udemy|alura|livraria', 'Educação', 'out'],
  ['cinema|ingresso|bar |cervejaria|steam|playstation|xbox|viagem|hotel|airbnb|booking|decolar', 'Lazer', 'out'],
  ['amazon|mercado livre|mercadolivre|shopee|magalu|magazine luiza|americanas|shein|aliexpress|renner|riachuelo', 'Compras', 'out'],
  ['iof|tarifa|anuidade|imposto|ipva|iptu|darf|multa', 'Impostos e taxas', 'out'],
].map(([pattern, tag, sign]) => ({ pattern, tag, sign: sign || 'any' }));

const ASSET_CLASSES = {
  'Ações': { color: '#2563eb', quote: 'brapi' },
  'FIIs': { color: '#16a34a', quote: 'brapi' },
  'ETFs': { color: '#0ea5e9', quote: 'brapi' },
  'BDRs': { color: '#8b5cf6', quote: 'brapi' },
  'Renda Fixa': { color: '#f59e0b', manual: true },
  'Tesouro Direto': { color: '#14b8a6', manual: true },
  'Cripto': { color: '#f97316', quote: 'coingecko' },
  'Exterior': { color: '#ec4899' },
  'Outros': { color: '#a3a3a3' },
};
Object.entries({ 'Ações': '📊', 'FIIs': '🏢', 'ETFs': '🧺', 'BDRs': '🌎', 'Renda Fixa': '🏦', 'Tesouro Direto': '🏛️', 'Cripto': '🪙', 'Exterior': '🌐', 'Outros': '📦' })
  .forEach(([k, v]) => { ASSET_CLASSES[k].icon = v; });
const COINGECKO_IDS = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana', USDT: 'tether', USDC: 'usd-coin', ADA: 'cardano', XRP: 'ripple', BNB: 'binancecoin', DOGE: 'dogecoin', DOT: 'polkadot', LINK: 'chainlink', MATIC: 'matic-network', AVAX: 'avalanche-2' };

function inferClass(ticker) {
  const t = ticker.toUpperCase();
  if (COINGECKO_IDS[t]) return 'Cripto';
  if (/^[A-Z]{4}11$/.test(t)) return 'FIIs';
  if (/^[A-Z]{4}3[2-5]$/.test(t)) return 'BDRs';
  if (/^[A-Z]{4}\d{1,2}$/.test(t)) return 'Ações';
  if (/TESOURO|SELIC|IPCA|PREFIXADO/.test(t)) return 'Tesouro Direto';
  if (/CDB|LCI|LCA|CRI|CRA|DEBENTURE/.test(t)) return 'Renda Fixa';
  return 'Outros';
}

/* =================================================================== state */
const STORAGE_KEY = 'minhas-financas.v1';
function defaultState() {
  return {
    version: 1,
    settings: { name: '', brapiToken: '' },
    accounts: [{ id: 'conta', name: 'Conta corrente', type: 'corrente', initialBalance: 0, color: '#7c3aed' }],
    tags: DEFAULT_TAGS.map(t => ({ ...t })),
    rules: DEFAULT_RULES.map(r => ({ id: uid(), ...r })),
    recurring: [],
    budgets: {},
    debts: [],
    transactions: [],
    investments: { assets: [], operations: [], history: {} },
  };
}
function normalizeState(s) {
  const d = defaultState();
  const merged = { ...d, ...s, settings: { ...d.settings, ...s.settings }, investments: { ...d.investments, ...s.investments }, budgets: s.budgets || {}, debts: s.debts || [], meta: s.meta || {} };
  for (const t of merged.tags) {
    const def = DEFAULT_TAGS.find(x => x.name === t.name);
    if (t.group === undefined) t.group = def?.group || '';
    if (t.icon === undefined) t.icon = def?.icon || '🏷️';
  }
  return merged;
}
function loadState() {
  try { const raw = localStorage.getItem(STORAGE_KEY); return raw ? normalizeState(JSON.parse(raw)) : defaultState(); }
  catch (e) { console.error(e); return defaultState(); }
}
let state = loadState();
// (o cache de cotações é aplicado logo após a definição de applyQuoteCache)
function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
  catch (e) { toast('Não foi possível salvar no navegador: ' + e.message); }
}
/** Grava uma alteração feita pelo usuário e agenda o envio para a nuvem (se a sincronização estiver ativa). */
function save() {
  state.meta = { ...(state.meta || {}), updatedAt: Date.now() };
  persist();
  Sync.schedulePush();
}
const hasData = st => !!(st.transactions.length || st.investments.operations.length);

const ui = {
  view: 'overview',
  month: currentMonth(),
  donutMode: 'out',
  filters: { month: currentMonth(), q: '', tag: '__all', account: '__all', day: '' },
  pendingImport: null,
  pendingInvestImport: null,
};

/* =========================================================== domain logic */
const tagInfo = name => state.tags.find(t => t.name === name) || (name ? { name, color: '#a3a3a3' } : NO_TAG);
const account = id => state.accounts.find(a => a.id === id);
const isExcluded = t => t.ignore || tagInfo(t.tag).exclude;

function suggestTag(description, amount) {
  const n = norm(description);
  for (const r of state.rules) {
    if (r.sign === 'in' && amount < 0) continue;
    if (r.sign === 'out' && amount > 0) continue;
    const alts = norm(r.pattern).split('|').map(s => s.trim()).filter(Boolean);
    if (alts.some(a => n.includes(a))) return r.tag;
  }
  return '';
}

function monthTx(ym) { return state.transactions.filter(t => t.date.startsWith(ym)); }

function accountBalance(a, end = todayISO()) {
  return (+a.initialBalance || 0) + sum(state.transactions.filter(t => t.accountId === a.id && t.date <= end), t => t.amount);
}
const accountColor = a => a.color || { corrente: '#7c3aed', cartao: '#e11d48', corretora: '#2563eb' }[a.type] || '#64748b';
const initials = name => String(name).split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join('').toUpperCase();
// Selos dos bancos: cor e sigla de cada instituição (não usa os logotipos oficiais; o usuário pode enviar uma imagem própria)
const BANKS = [
  ['nubank', 'Nubank', /\bnu(bank|conta)?\b/i, '#820ad1', '#fff', 'nu'],
  ['itau', 'Itaú', /ita[uú]/i, '#ec7000', '#fff', 'itaú'],
  ['bb', 'Banco do Brasil', /banco do brasil|\bbb\b/i, '#fcfc30', '#0038a8', 'BB'],
  ['bradesco', 'Bradesco', /bradesco/i, '#cc092f', '#fff', 'brad'],
  ['santander', 'Santander', /santander/i, '#ec0000', '#fff', 'sant'],
  ['caixa', 'Caixa', /caixa|\bcef\b/i, '#005ca9', '#f39200', 'CX'],
  ['inter', 'Inter', /\binter\b/i, '#ff7a00', '#fff', 'inter'],
  ['c6', 'C6 Bank', /\bc6\b/i, '#242424', '#fff', 'C6'],
  ['next', 'Next', /\bnext\b/i, '#00e45a', '#0b0b0b', 'next'],
  ['picpay', 'PicPay', /picpay/i, '#21c25e', '#fff', 'pic'],
  ['mercadopago', 'Mercado Pago', /mercado ?pago/i, '#00b1ea', '#fff', 'MP'],
  ['pagbank', 'PagBank', /pag ?(bank|seguro)/i, '#1bb99a', '#fff', 'pag'],
  ['neon', 'Neon', /\bneon\b/i, '#00e5ff', '#0b1f3a', 'neon'],
  ['will', 'Will Bank', /will ?bank/i, '#ffd900', '#111', 'will'],
  ['original', 'Banco Original', /original/i, '#00a857', '#fff', 'orig'],
  ['sicoob', 'Sicoob', /sicoob/i, '#003641', '#7db61c', 'scb'],
  ['sicredi', 'Sicredi', /sicredi/i, '#3fa110', '#fff', 'scr'],
  ['safra', 'Safra', /safra/i, '#002d5a', '#c9a14a', 'safra'],
  ['xp', 'XP Investimentos', /\bxp\b/i, '#111111', '#fff', 'XP'],
  ['btg', 'BTG Pactual', /btg/i, '#0d2c54', '#fff', 'btg'],
  ['rico', 'Rico', /\brico\b/i, '#ff5a00', '#fff', 'rico'],
  ['clear', 'Clear', /\bclear\b/i, '#00a3e0', '#fff', 'clr'],
  ['avenue', 'Avenue', /avenue/i, '#0a2240', '#fff', 'av'],
  ['binance', 'Binance', /binance/i, '#f3ba2f', '#111', 'bnb'],
].map(([id, name, match, bg, fg, label]) => ({ id, name, match, bg, fg, label }));
function bankOf(a) {
  if (a.bank === 'none') return null;
  if (a.bank) return BANKS.find(b => b.id === a.bank) || null;
  return BANKS.find(b => b.match.test(a.name || '')) || null;
}
function avatar(a, cls = '') {
  if (a.logo && /^data:image\//.test(a.logo)) return `<img class="avatar ${cls}" src="${esc(a.logo)}" alt="">`;
  const b = bankOf(a);
  if (b) return `<span class="avatar bank ${cls}" title="${esc(b.name)}" style="background:${b.bg};color:${b.fg};font-size:${b.label.length <= 2 ? 13 : b.label.length <= 4 ? 11 : 9}px">${esc(b.label)}</span>`;
  return `<span class="avatar ${cls}" style="background:${accountColor(a)}">${esc(initials(a.name))}</span>`;
}
function resizeImage(file, size = 96) {
  return new Promise((res, rej) => {
    const img = new Image(); const url = URL.createObjectURL(file);
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = c.height = size;
      const m = Math.min(img.width, img.height);
      c.getContext('2d').drawImage(img, (img.width - m) / 2, (img.height - m) / 2, m, m, 0, 0, size, size);
      URL.revokeObjectURL(url); res(c.toDataURL('image/png'));
    };
    img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('Não consegui ler essa imagem.')); };
    img.src = url;
  });
}

function spentByTag(ym, tag) {
  return -sum(monthTx(ym).filter(t => !isExcluded(t) && t.amount < 0 && (t.tag || '') === tag), t => t.amount);
}
function avgSpent(tag, ym, n = 3) {
  return sum(Array.from({ length: n }, (_, i) => spentByTag(addMonths(ym, -i - 1), tag))) / n;
}
function balanceGroups(ym) {
  const g = { essencial: 0, livre: 0, investimento: 0, sem: 0 };
  for (const t of monthTx(ym)) {
    if (isExcluded(t) || t.amount >= 0) continue;
    g[(t.tag && tagInfo(t.tag).group) || 'sem'] += -t.amount;
  }
  return g;
}

function monthSummary(ym) {
  const txs = monthTx(ym).filter(t => !isExcluded(t));
  const receitas = sum(txs.filter(t => t.amount > 0), t => t.amount);
  const despesas = -sum(txs.filter(t => t.amount < 0), t => t.amount);
  const end = `${ym}-${pad(daysInMonth(ym))}`;
  const contas = state.accounts.filter(a => a.type === 'corrente');
  const ids = new Set(contas.map(a => a.id));
  const saldo = sum(contas, a => +a.initialBalance || 0) +
    sum(state.transactions.filter(t => ids.has(t.accountId) && t.date <= end), t => t.amount);
  return { receitas, despesas, resultado: receitas - despesas, saldo, count: txs.length };
}

function byTag(ym, mode) {
  const txs = monthTx(ym).filter(t => !isExcluded(t) && (mode === 'out' ? t.amount < 0 : t.amount > 0));
  const map = new Map();
  for (const t of txs) map.set(t.tag || '', (map.get(t.tag || '') || 0) + Math.abs(t.amount));
  const items = [...map].map(([tag, value]) => ({ tag, label: tagInfo(tag).name, color: tagInfo(tag).color, icon: tagInfo(tag).icon, value }));
  items.sort((a, b) => b.value - a.value);
  return { items, total: sum(items, i => i.value) };
}

function runRecurring() {
  const today = todayISO(); let added = 0;
  for (const r of state.recurring) {
    r.generated = r.generated || [];
    for (let m = r.startMonth; m <= currentMonth() && (!r.endMonth || m <= r.endMonth); m = addMonths(m, 1)) {
      if (r.generated.includes(m)) continue;
      const date = `${m}-${pad(Math.min(+r.day || 1, daysInMonth(m)))}`;
      if (date > today) continue;
      const amount = r.debtId ? -debtPaymentFor(r.debtId, m) : +r.amount;
      if (r.debtId && !amount) { r.generated.push(m); continue; }
      state.transactions.push({ id: uid(), date, description: r.description, amount, tag: r.tag, accountId: r.accountId, source: 'recorrente', externalId: `rec:${r.id}:${m}` });
      r.generated.push(m); added++;
    }
  }
  if (added) save();
  return added;
}

/* ---- investimentos ---- */
function portfolio() {
  const pos = new Map(state.investments.assets.map(a => [a.id, { asset: a, qty: 0, cost: 0, realized: 0, dividends: 0, invested: 0, firstDate: null, div12: 0 }]));
  const ops = [...state.investments.operations].sort((a, b) => a.date.localeCompare(b.date));
  for (const op of ops) {
    const p = pos.get(op.assetId); if (!p) continue;
    const qty = +op.qty || 0, price = +op.price || 0, fees = +op.fees || 0;
    if (op.type === 'compra') { p.qty += qty; p.cost += qty * price + fees; p.invested += qty * price + fees; if (!p.firstDate) p.firstDate = op.date; }
    else if (op.type === 'venda') {
      const avg = p.qty ? p.cost / p.qty : 0;
      const q = Math.min(qty, p.qty);
      p.realized += q * price - fees - avg * q;
      p.cost -= avg * q; p.qty -= q;
    } else if (op.type === 'provento') { p.dividends += qty * price; if (op.date >= addMonths(currentMonth(), -12) + '-01') p.div12 += qty * price; }
  }
  const list = [...pos.values()].map(p => {
    const a = p.asset;
    p.auto = !!indexRate(a);
    p.market = +a.tdQty > 0 && +a.tdPrice > 0;
    const avg = p.qty > 1e-9 ? p.cost / p.qty : 0;
    const price = +a.currentPrice || avg;
    const manual = a.manualValue !== undefined && a.manualValue !== null && a.manualValue !== '';
    const auto = p.qty > 1e-9 ? indexedValue(a) : null;
    const td = +a.tdQty > 0 && +a.tdPrice > 0 ? +a.tdQty * +a.tdPrice : null;
    const value = p.qty > 1e-9 ? (td != null ? td : auto != null ? auto : manual ? +a.manualValue : p.qty * price) : 0;
    return { ...p, avg, price, value, result: value - p.cost, resultPct: p.cost ? (value - p.cost) / p.cost : 0 };
  });
  const totals = {
    value: sum(list, p => p.value), cost: sum(list, p => p.cost),
    realized: sum(list, p => p.realized), dividends: sum(list, p => p.dividends),
  };
  totals.result = totals.value - totals.cost;
  totals.resultPct = totals.cost ? totals.result / totals.cost : 0;
  return { list, totals };
}

function recordSnapshot() {
  state.investments.history = state.investments.history || {};
  state.investments.history[currentMonth()] = round2(portfolio().totals.value);
}

function aportesByMonth(months) {
  return months.map(m => sum(state.investments.operations.filter(o => o.date.startsWith(m) && o.type !== 'provento'),
    o => (o.type === 'compra' ? 1 : -1) * ((+o.qty || 0) * (+o.price || 0)) + (o.type === 'compra' ? (+o.fees || 0) : 0)));
}

function findOrCreateAsset(ticker, cls) {
  ticker = ticker.trim().toUpperCase();
  let a = state.investments.assets.find(x => x.ticker === ticker);
  if (!a) {
    cls = cls && cls !== 'auto' ? cls : inferClass(ticker);
    a = { id: uid(), ticker, class: cls, currentPrice: null, manualValue: null, quoteId: cls === 'Cripto' ? (COINGECKO_IDS[ticker] || ticker.toLowerCase()) : '' };
    state.investments.assets.push(a);
  }
  return a;
}

/** Taxa anual (%) de um título indexado: % do CDI, prefixado ou IPCA + taxa. */
function indexRate(a) {
  const r = +a.idxRate;
  if (!a.indexer || !(r > 0)) return null;
  const mk = typeof market === 'undefined' ? {} : market;
  const cdi = +state.settings.cdi || mk.cdi;
  if (a.indexer === 'cdi') return cdi ? (cdi * r) / 100 : null;
  if (a.indexer === 'pre') return r;
  if (a.indexer === 'ipca') return mk.ipca12 != null ? ((1 + mk.ipca12 / 100) * (1 + r / 100) - 1) * 100 : null;
  return null;
}
/** Valor bruto estimado: cada aporte rende pela taxa desde a data; resgates saem do saldo. */
function indexedValue(a) {
  const rate = indexRate(a);
  if (rate == null) return null;
  const today = new Date(todayISO() + 'T12:00:00').getTime();
  const ops = state.investments.operations.filter(o => o.assetId === a.id && o.type !== 'provento');
  let v = 0;
  for (const o of ops) {
    const years = Math.max(0, (today - new Date(o.date + 'T12:00:00').getTime()) / (365.25 * 864e5));
    const amt = (+o.qty || 0) * (+o.price || 0);
    v += (o.type === 'compra' ? 1 : -1) * amt * Math.pow(1 + rate / 100, years);
  }
  return Math.max(0, v);
}
setTimeout(() => { applyQuoteCache(); }, 0);
const INDEXERS = { '': 'Manual (eu informo o valor)', cdi: '% do CDI', pre: 'Prefixado (% a.a.)', ipca: 'IPCA + taxa (% a.a.)' };

/* ---------- cotações: cache local (não conflita com a sincronização) e atualização automática ---------- */
const QUOTES_KEY = 'minhas-financas.quotes';
let quoteCache = (() => { try { return JSON.parse(localStorage.getItem(QUOTES_KEY)) || {}; } catch { return {}; } })();
function applyQuoteCache() {
  for (const a of state.investments.assets) {
    const q = quoteCache[a.ticker];
    if (q && (!a.priceUpdatedAt || q.at > a.priceUpdatedAt)) { a.currentPrice = q.price; a.priceUpdatedAt = q.at; if (q.high52) { a.high52 = q.high52; a.low52 = q.low52; } }
    const t = a.tdSymbol && quoteCache['TD:' + a.tdSymbol];
    if (t && (!a.tdPriceAt || t.at > a.tdPriceAt)) { a.tdPrice = t.price; a.tdPriceAt = t.at; }
  }
}
function isMarketHours() { const d = new Date(), h = d.getHours(), w = d.getDay(); return w >= 1 && w <= 5 && h >= 10 && h < 18; }
let quotesBusy = false, quoteHintShown = false;
async function autoQuotes(force = false) {
  if (quotesBusy || !navigator.onLine) return;
  const last = state.investments.quotesAt || 0;
  const age = Date.now() - last;
  const hasB3 = state.investments.assets.some(a => ['Ações', 'FIIs', 'ETFs', 'BDRs'].includes(a.class));
  const hasCrypto = state.investments.assets.some(a => a.class === 'Cripto');
  const hasTD = state.investments.assets.some(a => a.tdSymbol);
  // B3: no pregão a cada 20 min; fora dele, uma vez ao abrir (fechamento). Cripto: 24h, a cada 20 min.
  const due = force || age > 20 * 60e3 && (((hasB3 || hasTD) && (isMarketHours() || age > 6 * 3600e3)) || hasCrypto);
  if (due) await updateQuotes({ silent: true });
}
function safeRender() {
  const el = document.activeElement;
  if ($('#modal').open || (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))) { ui.renderPending = true; return; }
  render();
}
document.addEventListener('focusout', () => { if (ui.renderPending) setTimeout(() => { const el = document.activeElement; if (ui.renderPending && !(el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) && !$('#modal').open) { ui.renderPending = false; render(); } }, 300); });

async function updateQuotes({ silent = false } = {}) {
  const assets = state.investments.assets.filter(a => ASSET_CLASSES[a.class]?.quote);
  const tds = state.investments.assets.filter(a => a.tdSymbol);
  if (!assets.length && !tds.length) { if (!silent) toast('Nenhum ativo com cotação automática (Ações, FIIs, ETFs, BDRs, Cripto).'); return; }
  if (!silent) toast('Atualizando cotações…');
  quotesBusy = true;
  const token = state.settings.brapiToken?.trim();
  const ok = [], fail = [];
  const crypto = assets.filter(a => a.class === 'Cripto');
  const b3 = assets.filter(a => a.class !== 'Cripto');
  const jobs = b3.map(async a => {
    const url = `https://brapi.dev/api/quote/${encodeURIComponent(a.ticker)}` + (token ? `?token=${encodeURIComponent(token)}` : '');
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    const q0 = j?.results?.[0] || {}; const price = q0.regularMarketPrice;
    if (!price) throw new Error('sem preço');
    a.currentPrice = price; a.priceUpdatedAt = new Date().toISOString(); ok.push(a.ticker);
    if (q0.fiftyTwoWeekHigh) { a.high52 = q0.fiftyTwoWeekHigh; a.low52 = q0.fiftyTwoWeekLow; }
    quoteCache[a.ticker] = { price, at: a.priceUpdatedAt, high52: a.high52, low52: a.low52 };
  }).map((p, i) => p.catch(() => fail.push(b3[i].ticker)));
  if (tds.length) {
    // Preço unitário (PU) de mercado dos títulos do Tesouro. Aceita variações de nome de campo na resposta.
    jobs.push((async () => {
      try {
        const syms = [...new Set(tds.map(a => a.tdSymbol.trim().toLowerCase()))].slice(0, 20);
        const r = await fetch(`https://brapi.dev/api/v2/treasury/indicators?symbols=${syms.join(',')}` + (token ? `&token=${encodeURIComponent(token)}` : ''));
        if (!r.ok) throw new Error(r.status);
        const j = await r.json();
        const items = [j.results, j.indicators, j.data, j.bonds, Array.isArray(j) ? j : null].find(Array.isArray) || [];
        const num = x => (typeof x === 'number' ? x : parseFloat(String(x ?? '').replace(',', '.')));
        const pick = it => ['sellUnitPrice', 'unitPriceSell', 'sellPrice', 'redemptionUnitPrice', 'redemptionPrice', 'puVenda', 'unitPrice', 'price', 'pu', 'buyUnitPrice', 'buyPrice']
          .map(k => num(it[k] ?? it.prices?.[k] ?? it.price?.[k])).find(v => v > 0);
        for (const a of tds) {
          const it = items.find(x => String(x.symbol || x.code || x.id || '').toLowerCase() === a.tdSymbol.trim().toLowerCase());
          const pu = it && pick(it);
          if (pu) { a.tdPrice = pu; a.tdPriceAt = new Date().toISOString(); quoteCache['TD:' + a.tdSymbol] = { price: pu, at: a.tdPriceAt }; ok.push(a.ticker); } else fail.push(a.ticker);
        }
      } catch { tds.forEach(a => fail.push(a.ticker)); }
    })());
  }
  if (crypto.length) {
    jobs.push((async () => {
      const ids = crypto.map(a => a.quoteId || COINGECKO_IDS[a.ticker] || a.ticker.toLowerCase());
      try {
        const r = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${ids.join(',')}&vs_currencies=brl`);
        const j = await r.json();
        crypto.forEach((a, i) => {
          const p = j?.[ids[i]]?.brl;
          if (p) { a.currentPrice = p; a.priceUpdatedAt = new Date().toISOString(); quoteCache[a.ticker] = { price: p, at: a.priceUpdatedAt }; ok.push(a.ticker); } else fail.push(a.ticker);
        });
      } catch { crypto.forEach(a => fail.push(a.ticker)); }
    })());
  }
  await Promise.all(jobs);
  quotesBusy = false;
  // Preços ficam no cache local e no estado SEM marcar alteração: não disputam a sincronização com lançamentos do outro aparelho.
  try { localStorage.setItem(QUOTES_KEY, JSON.stringify(quoteCache)); } catch { /* */ }
  if (ok.length) { state.investments.quotesAt = Date.now(); recordSnapshot(); persist(); safeRender(); }
  let msg = `${ok.length} cotação(ões) atualizada(s).`;
  if (fail.length) msg += ` Falharam: ${fail.join(', ')}` + (token ? '' : ' — configure um token gratuito da brapi.dev em Configurações.');
  if (!silent) toast(msg, 6000);
  else if (fail.length && !token && !quoteHintShown) { quoteHintShown = true; toast('Algumas cotações não atualizaram sozinhas: crie um token gratuito em brapi.dev e cole em Configurações.', 8000); }
}

/* ================================================================== charts */
function donutSVG(items, total, top, bottom) {
  const R = 60, C = 2 * Math.PI * R; let off = 0;
  const segs = items.filter(i => i.value > 0).map(i => {
    const len = total ? (i.value / total) * C : 0;
    const s = `<circle r="${R}" cx="80" cy="80" fill="none" stroke="${i.color}" stroke-width="22" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 80 80)"><title>${esc(i.label)}: ${fmt(i.value)}</title></circle>`;
    off += len; return s;
  }).join('');
  return `<svg viewBox="0 0 160 160" class="donut" role="img">
    ${segs || `<circle r="${R}" cx="80" cy="80" fill="none" stroke="var(--line)" stroke-width="22"/>`}
    <text x="80" y="80" text-anchor="middle" class="donut-v">${esc(top)}</text>
    <text x="80" y="95" text-anchor="middle" class="donut-l">${esc(bottom)}</text></svg>`;
}

function legendHTML(items, total, action) {
  if (!items.length) return '<p class="muted">Sem dados no período.</p>';
  return `<ul class="legend">${items.map(i => `
    <li ${action ? `data-action="${action}" data-tag="${esc(i.tag)}"` : ''}>
      ${i.icon ? iconBadge(i.icon, i.color, 'sm') : `<span class="dot" style="background:${i.color}"></span>`}
      <span>${esc(i.label)}</span>
      <span class="val">${fmt(i.value)}</span>
      <span class="p">${pct(total ? i.value / total : 0)}</span>
    </li>`).join('')}</ul>`;
}

function barsSVG(labels, series, { height = 200 } = {}) {
  const W = 640, H = height, L = 56, B = 24, T = 8;
  const all = series.flatMap(s => s.values);
  const max = Math.max(1, ...all.map(v => Math.max(0, v)));
  const min = Math.min(0, ...all);
  const range = max - min;
  const y = v => T + (H - T - B) * (1 - (v - min) / range);
  const gw = (W - L) / labels.length;
  const bw = Math.min(28, (gw * 0.7) / series.length);
  const ticks = 4;
  let g = '';
  for (let i = 0; i <= ticks; i++) {
    const v = min + (range * i) / ticks;
    g += `<line class="grid-line" x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtShort(v).replace('R$ ', '')}</text>`;
  }
  labels.forEach((lab, i) => {
    const cx = L + gw * i + gw / 2;
    series.forEach((s, j) => {
      const v = s.values[i] || 0;
      const x = cx - (bw * series.length) / 2 + j * bw;
      const y0 = y(Math.max(0, v)), h = Math.abs(y(v) - y(0));
      g += `<rect x="${x + 1}" y="${y0}" width="${bw - 2}" height="${Math.max(h, 0.5)}" rx="3" fill="${s.color}"><title>${esc(s.label)} — ${esc(lab)}: ${fmt(v)}</title></rect>`;
    });
    g += `<text x="${cx}" y="${H - 6}" text-anchor="middle">${esc(lab)}</text>`;
  });
  const legend = series.length > 1 ? `<div class="chart-legend">${series.map(s => `<span><span class="dot" style="background:${s.color}"></span> ${esc(s.label)}</span>`).join('')}</div>` : '';
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${g}</svg>${legend}`;
}

function lineSVG(labels, values, color = 'var(--blue)') {
  const W = 640, H = 200, L = 56, B = 24, T = 10;
  const max = Math.max(1, ...values), min = Math.min(...values, max * 0.8);
  const range = max - min || 1;
  const x = i => L + ((W - L - 10) * i) / Math.max(1, labels.length - 1);
  const y = v => T + (H - T - B) * (1 - (v - min) / range);
  let g = '';
  for (let i = 0; i <= 4; i++) { const v = min + (range * i) / 4; g += `<line class="grid-line" x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtShort(v).replace('R$ ', '')}</text>`; }
  const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(' ');
  g += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>`;
  values.forEach((v, i) => { g += `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${color}"><title>${esc(labels[i])}: ${fmt(v)}</title></circle><text x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(labels[i])}</text>`; });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

/* ================================================================== views */
function greeting() {
  const h = new Date().getHours();
  const g = h < 12 ? 'Bom dia' : h < 18 ? 'Boa tarde' : 'Boa noite';
  return state.settings.name ? `${g}, ${esc(state.settings.name)}!` : `${g}!`;
}

function monthPicker() {
  return `<div class="month-picker">
    <button class="icon-btn" data-action="month-prev" title="Mês anterior">‹</button>
    <span>📅 ${monthLabel(ui.month)}</span>
    <button class="icon-btn" data-action="month-next" title="Próximo mês">›</button>
  </div>`;
}

function emptyState() {
  return `<div class="card empty-state">
    <h2>Comece a acompanhar seu dinheiro</h2>
    <p>Importe o extrato do seu banco (OFX ou CSV) — as transações são categorizadas automaticamente pelas regras.
      Ou carregue dados de exemplo para conhecer o painel.</p>
    <button class="btn primary" data-action="nav" data-view="import">Importar extrato</button>
    <button class="btn" data-action="load-demo">Carregar dados de exemplo</button>
  </div>`;
}

function viewOverview() {
  const s = monthSummary(ui.month);
  const head = `<div class="page-head"><div><h1>${greeting()}</h1><p>O que está acontecendo hoje?</p></div>
    <div class="toolbar">${quickActions()}${monthPicker()}</div></div>` + onboardingCard();
  if (!state.transactions.length && !state.investments.operations.length) return head + emptyState();

  const kpis = `<div class="grid kpis">
    <div class="card kpi"><div class="label">Resultado do período</div><div class="value ${s.resultado >= 0 ? '' : 'neg'}">${fmt(s.resultado)}</div><div class="hint">receitas − despesas</div></div>
    <div class="card kpi"><div class="label">Receitas</div><div class="value pos">${fmt(s.receitas)}</div></div>
    <div class="card kpi"><div class="label">Despesas</div><div class="value neg">${fmt(s.despesas)}</div></div>
    <div class="card kpi"><div class="label">Saldo em conta</div><div class="value">${fmt(s.saldo)}</div><div class="hint">até o fim de ${monthLabel(ui.month)}</div></div>
  </div>`;

  // heatmap de saídas por dia
  const days = daysInMonth(ui.month);
  const out = Array(days + 1).fill(0);
  for (const t of monthTx(ui.month)) if (t.amount < 0 && !isExcluded(t)) out[+t.date.slice(8, 10)] += -t.amount;
  const max = Math.max(...out);
  const withOut = out.filter(v => v > 0).length;
  const [y, m] = ui.month.split('-').map(Number);
  const first = new Date(y, m - 1, 1).getDay();
  const today = todayISO();
  let cells = ['DOM', 'SEG', 'TER', 'QUA', 'QUI', 'SEX', 'SÁB'].map(d => `<div class="dow">${d}</div>`).join('');
  cells += '<div class="day empty"></div>'.repeat(first);
  for (let d = 1; d <= days; d++) {
    const iso = `${ui.month}-${pad(d)}`;
    const v = out[d];
    const ratio = max ? v / max : 0;
    const bg = v > 0 ? `background: rgba(var(--heat), ${0.12 + ratio * 0.78})` : '';
    cells += `<div class="day ${iso > today ? 'future' : ''} ${ratio > 0.45 ? 'hot' : ''}" style="${bg}" data-action="goto-day" data-day="${iso}" title="${fmtDate(iso)}: ${fmt(v)}">
      <span class="n">${d}</span><span class="v">${v > 0 ? fmtShort(v) : ''}</span></div>`;
  }
  const heat = `<div class="card">
    <h2>Saídas por dia</h2><div class="sub">${monthLabel(ui.month)} · quanto saiu em cada dia</div>
    <div class="big-number">${fmt(s.despesas)}</div><div class="muted small">em ${withOut} dias com saída</div>
    <div class="cal">${cells}</div></div>`;

  const { items, total } = byTag(ui.month, ui.donutMode);
  const tags = `<div class="card">
    <div class="card-head"><h2>Por tags</h2>
      <div class="seg"><button class="${ui.donutMode === 'out' ? 'active' : ''}" data-action="donut" data-mode="out">Despesas</button><button class="${ui.donutMode === 'in' ? 'active' : ''}" data-action="donut" data-mode="in">Receitas</button></div>
    </div>
    <div class="donut-wrap mt">${donutSVG(items, total, fmt(total), 'total no mês')}${legendHTML(items, total, 'goto-tag')}</div></div>`;

  const months = Array.from({ length: 6 }, (_, i) => addMonths(ui.month, i - 5));
  const sums = months.map(monthSummary);
  const evo = `<div class="card"><h2>Receitas × despesas</h2><div class="sub">últimos 6 meses</div>
    ${barsSVG(months.map(monthShort), [
      { label: 'Receitas', color: 'var(--green)', values: sums.map(x => x.receitas) },
      { label: 'Despesas', color: 'var(--red)', values: sums.map(x => x.despesas) },
    ])}</div>`;

  const pf = portfolio().totals;
  const top = monthTx(ui.month).filter(t => t.amount < 0 && !isExcluded(t)).sort((a, b) => a.amount - b.amount).slice(0, 6);
  const side = `<div class="card">
    <div class="card-head"><div><h2>Investimentos</h2><div class="sub">patrimônio investido hoje</div></div><button class="btn small" data-action="nav" data-view="investments">Ver carteira →</button></div>
    <div class="big-number">${fmt(pf.value)}</div>
    <div class="small ${pf.result >= 0 ? 'pos' : 'neg'}">${pf.result >= 0 ? '▲' : '▼'} ${fmt(pf.result)} (${pct(pf.resultPct)}) sobre ${fmt(pf.cost)} aplicados</div>
    <h2 class="mt">Maiores gastos do mês</h2>
    ${top.length ? top.map(t => `<div class="list-row">${tagIcon(t.tag)}<span class="grow">${esc(t.description)}<br><span class="muted small">${fmtDate(t.date)} · ${esc(tagInfo(t.tag).name)}</span></span><span class="neg">${fmt(t.amount)}</span></div>`).join('') : '<p class="muted">Nenhum gasto no mês.</p>'}
  </div>`;

  return head + kpis + `<div class="grid two">${heat}${tags}</div>
    <div class="grid two">${accountsCard()}<div class="stack">${goalsCard()}${balanceCard()}</div></div>
    <div class="grid two">${evo}${side}</div>`;
}

function quickActions(menu = false) {
  return `<div class="quick ${menu ? 'quick-grid' : ''}">
    <button class="qa" data-action="quick-tx" data-kind="out"><span class="qa-i neg">−</span>Despesa</button>
    <button class="qa" data-action="quick-tx" data-kind="in"><span class="qa-i pos">+</span>Receita</button>
    <button class="qa" data-action="transfer"><span class="qa-i">⇄</span>Transferência</button>
    <button class="qa" data-action="add-op"><span class="qa-i">↗</span>Investimento</button>
    ${menu ? '<button class="qa" data-action="nav" data-view="import"><span class="qa-i">⬆</span>Importar extrato</button>' : ''}
  </div>`;
}

function onboardingSteps() {
  return [
    { label: 'Cadastrar suas contas e cartões', done: state.accounts.length > 1 || state.accounts.some(a => +a.initialBalance), go: 'settings' },
    { label: 'Importar um extrato do banco', done: state.transactions.some(t => t.source === 'importado'), go: 'import' },
    { label: 'Categorizar todas as transações', done: state.transactions.length > 0 && !state.transactions.some(t => !t.tag), go: 'transactions' },
    { label: 'Registrar seus investimentos', done: state.investments.operations.length > 0, go: 'investments' },
    { label: 'Definir metas do mês', done: Object.values(state.budgets || {}).some(v => +v > 0), go: 'budgets' },
  ];
}
function onboardingCard() {
  if (prefs.hideOnboarding) return '';
  const st = onboardingSteps(); const n = st.filter(x => x.done).length;
  if (n === st.length) return '';
  const next = st.find(x => !x.done);
  return `<div class="card onboarding"><div class="ob-icon">✨</div>
    <div class="grow"><strong>Primeiros passos</strong>
      <div class="small muted"><strong class="pos">${n}</strong> de ${st.length} tarefas completas · próximo: ${esc(next.label)}</div>
      <div class="steps">${st.map(x => `<span class="${x.done ? 'done' : ''}" title="${esc(x.label)}"></span>`).join('')}</div></div>
    <button class="btn soft" data-action="onboarding-go" data-go="${next.go}">Continuar</button>
    <button class="icon-btn" data-action="hide-onboarding" title="Dispensar">✕</button></div>`;
}

function accountsCard() {
  const banks = state.accounts.filter(a => a.type !== 'cartao');
  const cards = state.accounts.filter(a => a.type === 'cartao');
  const total = sum(banks, a => accountBalance(a));
  const cardHTML = a => {
    const fat = Math.max(0, -accountBalance(a)); const lim = +a.limit || 0;
    return `<div class="cc">
      <div class="list-row">${avatar(a)}<span class="grow"><strong>${esc(a.name)}</strong><br><span class="muted small">Cartão de crédito</span></span>
        <button class="btn small soft" data-action="goto-account" data-id="${a.id}">ver fatura</button></div>
      <div class="cc-nums"><div><span class="muted small">Disponível</span><br><strong>${lim ? fmt(lim - fat) : '—'}</strong></div>
        <div><span class="muted small">Fatura atual</span><br><strong class="neg">${fmt(fat)}</strong></div></div>
      ${lim ? `<div class="bar cc-bar"><span style="width:${Math.min(100, (fat / lim) * 100)}%;background:var(--red)"></span></div>` : ''}</div>`;
  };
  return `<div class="card">
    <div class="card-head"><div><h2>Minhas contas</h2><div class="sub">saldo geral hoje</div></div><button class="btn small" data-action="adjust-balances">Ajustar saldos</button></div>
    <div class="big-number">${fmt(total)}</div>
    ${banks.map(a => `<div class="list-row clickable" data-action="goto-account" data-id="${a.id}">${avatar(a)}<span class="grow"><strong>${esc(a.name)}</strong><br><span class="muted small">${ACC_TYPES[a.type]}</span></span><span class="acc-val">${fmt(accountBalance(a))}</span></div>`).join('')}
    ${cards.length ? `<h2 class="mt">Cartões de crédito</h2>${cards.map(cardHTML).join('')}` : `<p class="muted small mt">Cadastre seus cartões em Configurações para acompanhar limite e fatura.</p>`}
    ${debtsMini()}
  </div>`;
}

function goalsCard() {
  const entries = Object.entries(state.budgets || {}).filter(([, v]) => +v > 0)
    .map(([tag, lim]) => ({ tag, lim: +lim, spent: spentByTag(ui.month, tag) }))
    .sort((a, b) => b.spent / b.lim - a.spent / a.lim);
  const head = `<div class="card-head"><div><h2>Metas de ${MONTHS[+ui.month.slice(5) - 1]}</h2><div class="sub">limite de gastos por tag</div></div>
    <button class="btn small" data-action="edit-budgets">${entries.length ? 'Editar' : 'Definir metas'}</button></div>`;
  if (!entries.length) return `<div class="card">${head}<p class="muted">Defina quanto quer gastar por categoria no mês — o app sugere valores pela sua média dos últimos 3 meses.</p></div>`;
  return `<div class="card">${head}${entries.map(({ tag, lim, spent }) => {
    const r = spent / lim;
    const color = r > 1 ? 'var(--red)' : r > 0.8 ? 'var(--amber)' : 'var(--green)';
    return `<div class="goal"><div class="goal-top"><span class="goal-name">${tagIcon(tag, 'sm')} ${esc(tag)}</span>
      <span class="small">${fmt(spent)} <span class="muted">de ${fmt(lim)}</span></span></div>
      <div class="bar"><span style="width:${Math.min(100, r * 100)}%;background:${color}"></span></div>
      <div class="small ${r > 1 ? 'neg' : 'muted'}">${r > 1 ? `estourou em ${fmt(spent - lim)}` : `restam ${fmt(lim - spent)}`}</div></div>`;
  }).join('')}</div>`;
}

function balanceCard() {
  const inc = monthSummary(ui.month).receitas; const g = balanceGroups(ui.month);
  const row = (label, value, share, min = false) => {
    const target = inc * share; const r = target ? value / target : 0;
    const color = min ? (r >= 1 ? 'var(--green)' : 'var(--amber)') : (r <= 0.9 ? 'var(--green)' : r <= 1 ? 'var(--amber)' : 'var(--red)');
    return `<div class="goal"><div class="goal-top"><span>${label}</span><strong class="small">${fmt(value)} <span class="muted">· ${pct(inc ? value / inc : 0)}</span></strong></div>
      <div class="bar"><span style="width:${Math.min(100, r * 100)}%;background:${color}"></span></div>
      <div class="small muted">${min ? 'Mínimo' : 'Limite'} recomendado: ${fmt(target)} (${share * 100}% da receita)</div></div>`;
  };
  return `<div class="card"><h2>Equilíbrio financeiro</h2><div class="sub">regra 50/30/20 sobre as receitas do mês · grupo de cada tag em Configurações</div>
    ${inc ? row('Gastos essenciais', g.essencial, 0.5) + row('Gastos não essenciais', g.livre, 0.3) + row('Investimentos', g.investimento, 0.2, true) : '<p class="muted">Sem receitas no mês.</p>'}
    ${g.sem ? `<p class="small muted">${fmt(g.sem)} em gastos sem tag (ou tag sem grupo) não entram na conta.</p>` : ''}</div>`;
}

function quickTxForm(out) {
  return `<div class="form-grid">
    <label>Valor (R$)<input type="number" step="0.01" min="0" name="value" required autofocus></label>
    <label>Data<input type="date" name="date" required value="${todayISO()}"></label>
    <label class="full">Descrição<input type="text" name="description" required placeholder="${out ? 'Mercado, Uber, farmácia…' : 'Salário, freela, reembolso…'}"></label>
    <label>Tag<select name="tag"><option value="">Automática (pelas regras)</option>${state.tags.map(t => `<option>${esc(t.name)}</option>`).join('')}</select></label>
    <label>Conta<select name="accountId">${accountOptions((out && state.accounts.find(a => a.type === 'cartao')?.id) || state.accounts[0]?.id)}</select></label>
  </div>`;
}
function transferForm() {
  const [a, b] = state.accounts;
  return `<div class="form-grid">
    <label>De<select name="from">${accountOptions(a?.id)}</select></label>
    <label>Para<select name="to">${accountOptions((b || a)?.id)}</select></label>
    <label>Valor (R$)<input type="number" step="0.01" min="0" name="value" required></label>
    <label>Data<input type="date" name="date" required value="${todayISO()}"></label>
  </div><p class="muted small">Transferências e pagamentos de fatura não contam como receita nem despesa — só movem saldo entre suas contas.</p>`;
}

function filteredTx() {
  const f = ui.filters; const q = norm(f.q);
  return state.transactions.filter(t =>
    (f.day ? t.date === f.day : (f.month === '__all' || t.date.startsWith(f.month))) &&
    (f.tag === '__all' || (t.tag || '') === f.tag) &&
    (f.account === '__all' || t.accountId === f.account) &&
    (!q || norm(t.description).includes(q))
  ).sort((a, b) => b.date.localeCompare(a.date) || a.description.localeCompare(b.description));
}

function tagOptions(selected, { all = false, none = true } = {}) {
  return (all ? `<option value="__all" ${selected === '__all' ? 'selected' : ''}>Todas as tags</option>` : '') +
    (none ? `<option value="" ${selected === '' ? 'selected' : ''}>Sem Tags</option>` : '') +
    state.tags.map(t => `<option ${t.name === selected ? 'selected' : ''}>${esc(t.name)}</option>`).join('');
}
function accountOptions(selected, all = false) {
  return (all ? `<option value="__all">Todas as contas</option>` : '') +
    state.accounts.map(a => `<option value="${a.id}" ${a.id === selected ? 'selected' : ''}>${esc(a.name)}</option>`).join('');
}

function viewTransactions() {
  const f = ui.filters;
  const list = filteredTx();
  const ins = sum(list.filter(t => t.amount > 0 && !isExcluded(t)), t => t.amount);
  const outs = -sum(list.filter(t => t.amount < 0 && !isExcluded(t)), t => t.amount);
  const untagged = state.transactions.filter(t => !t.tag).length;
  const monthOpts = [...new Set([currentMonth(), ...state.transactions.map(t => t.date.slice(0, 7))])].sort().reverse();
  const rows = list.slice(0, 500).map(t => `
    <tr class="${isExcluded(t) ? 'muted' : ''}">
      <td>${fmtDate(t.date)}</td>
      <td class="desc">${esc(t.description)}${t.source && t.source !== 'manual' ? ` <span class="muted small">· ${esc(t.source)}</span>` : ''}</td>
      <td class="muted"><div class="acc-cell">${account(t.accountId) ? avatar(account(t.accountId), 'xs') : ''}${esc(account(t.accountId)?.name || '—')}</div></td>
      <td><div class="tag-cell">${tagIcon(t.tag || '', 'sm')}<select class="tag-select" data-change="tx-tag" data-id="${t.id}">${tagOptions(t.tag || '')}</select></div></td>
      <td class="num ${t.amount >= 0 ? 'pos' : 'neg'}">${fmt(t.amount)}</td>
      <td><div class="row-actions">
        <button class="icon-btn" title="Criar regra a partir desta transação" data-action="rule-from-tx" data-id="${t.id}">⚡</button>
        <button class="icon-btn" title="Editar" data-action="edit-tx" data-id="${t.id}">✎</button>
        <button class="icon-btn" title="Excluir" data-action="del-tx" data-id="${t.id}">✕</button>
      </div></td>
    </tr>`).join('');
  return `<div class="page-head"><div><h1>Transações</h1><p>${list.length} transações · entradas <span class="pos">${fmt(ins)}</span> · saídas <span class="neg">${fmt(outs)}</span></p></div>
    <div class="toolbar"><button class="btn" data-action="add-tx">+ Nova transação</button>
    ${untagged ? `<button class="btn" data-action="apply-rules">⚡ Aplicar regras em ${untagged} sem tag</button>` : ''}</div></div>
    <div class="card">
      <div class="toolbar">
        <input type="search" placeholder="Buscar descrição…" value="${esc(f.q)}" data-input="filter-q">
        ${f.day ? `<span class="pill">Dia ${fmtDate(f.day)} <button class="icon-btn" data-action="clear-day">✕</button></span>` :
        `<select data-change="filter-month"><option value="__all">Todos os meses</option>${monthOpts.map(m => `<option value="${m}" ${m === f.month ? 'selected' : ''}>${monthLabel(m)}</option>`).join('')}</select>`}
        <select data-change="filter-tag">${tagOptions(f.tag, { all: true })}</select>
        <select data-change="filter-account">${accountOptions(f.account, true).replace(`value="${f.account}"`, `value="${f.account}" selected`)}</select>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Data</th><th>Descrição</th><th>Conta</th><th>Tag</th><th class="num">Valor</th><th></th></tr></thead>
        <tbody>${rows || '<tr><td colspan="6" class="muted">Nenhuma transação encontrada.</td></tr>'}</tbody>
      </table></div>
      ${list.length > 500 ? '<p class="muted small">Mostrando as 500 mais recentes — use os filtros.</p>' : ''}
    </div>`;
}

/* ---------- análise da carteira (sinais objetivos, sem previsão de mercado) ---------- */
const TAX = { // regras gerais de IR na venda (pessoa física) — confirmar com a Receita/contador
  'Ações': { rate: 0.15, exempt: 20000, note: 'ações: isento se as vendas de ações no mês somarem até R$ 20 mil' },
  'FIIs': { rate: 0.20, note: 'FIIs: 20% sobre o lucro, sem isenção' },
  'ETFs': { rate: 0.15, note: 'ETFs: 15% sobre o lucro, sem isenção' },
  'BDRs': { rate: 0.15, note: 'BDRs: 15% sobre o lucro, sem isenção' },
  'Cripto': { rate: 0.15, exempt: 35000, note: 'cripto: isento se as vendas no mês somarem até R$ 35 mil' },
  'Exterior': { rate: 0.15, note: 'exterior: 15% na declaração anual' },
  'Outros': { rate: 0.15, note: '' }, 'Renda Fixa': { rate: 0.15, note: 'renda fixa: IR regressivo retido na fonte' }, 'Tesouro Direto': { rate: 0.15, note: 'Tesouro: IR regressivo retido na fonte' },
};
const analysisCdi = () => +state.settings.cdi || market.cdi || 0;
function monthSales(cls, ym = currentMonth()) {
  return sum(state.investments.operations.filter(o => o.type === 'venda' && o.date.startsWith(ym) && state.investments.assets.find(a => a.id === o.assetId)?.class === cls), o => o.qty * o.price);
}
function assetSignals(p, total) {
  const a = p.asset, out = [], cdi = analysisCdi();
  const weight = total ? p.value / total : 0;
  const t = p.firstDate ? (Date.now() - new Date(p.firstDate + 'T12:00:00').getTime()) / (365.25 * 864e5) : 0;
  if (cdi && t >= 0.5 && p.cost > 0) {
    const ret = (p.value + p.dividends + p.realized - p.cost) / p.cost;
    const cdiRet = Math.pow(1 + cdi / 100, t) - 1;
    const gap = (cdiRet - ret) * p.cost;
    if (ret < cdiRet - 0.02) out.push({ lv: 'warn', k: 'cdi', t: `Rendeu ${pct(ret)} (com proventos) desde ${fmtDate(p.firstDate)}; o CDI rendeu ~${pct(cdiRet)} no período. Custo de oportunidade ≈ ${fmt(gap)}.` });
    else if (ret > cdiRet + 0.02) out.push({ lv: 'good', k: 'cdi', t: `Rendeu ${pct(ret)} desde ${fmtDate(p.firstDate)}, acima do CDI (~${pct(cdiRet)}).` });
  }
  if (weight > 0.15 && !ASSET_CLASSES[a.class]?.manual) out.push({ lv: 'warn', k: 'conc', t: `Representa ${pct(weight)} da carteira: concentração alta em um único ativo.` });
  if (+a.high52 > 0 && +a.low52 > 0 && p.price) {
    const dd = 1 - p.price / a.high52, pos = (p.price - a.low52) / Math.max(0.01, a.high52 - a.low52);
    if (dd > 0.2) out.push({ lv: 'info', k: '52', t: `Está ${pct(dd)} abaixo da máxima de 52 semanas (${fmt(a.high52)}). Queda pode ser oportunidade ou piora dos fundamentos — vale reavaliar a tese.` });
    else if (pos > 0.9) out.push({ lv: 'info', k: '52', t: `Perto da máxima de 52 semanas (${fmt(a.high52)}). Se passou do peso-alvo, é um bom momento para rebalancear.` });
  }
  if (a.class === 'FIIs' && cdi && p.value > 0 && p.firstDate && t >= 1) {
    const dy = p.div12 / p.value, cdiNet = (cdi * 0.85) / 100;
    if (dy < cdiNet - 0.02) out.push({ lv: 'warn', k: 'dy', t: `Rendimento 12m de ${pct(dy)} (isento), abaixo do CDI líquido (~${pct(cdiNet)}).` });
    else out.push({ lv: 'good', k: 'dy', t: `Rendimento 12m de ${pct(dy)} (isento) × CDI líquido ~${pct(cdiNet)}.` });
  }
  if (p.result < -1 && !ASSET_CLASSES[a.class]?.manual) out.push({ lv: 'info', k: 'loss', t: `Prejuízo não realizado de ${fmt(-p.result)}: se decidir vender, o prejuízo pode compensar IR de ganhos futuros na mesma categoria.` });
  return out;
}

function viewAnalysis() {
  const { list, totals } = portfolio();
  const active = list.filter(p => p.qty > 1e-9);
  const total = totals.value;
  const tg = state.investments.targets || {};
  const head = `<div class="page-head"><div><button class="btn small ghost" data-action="close-analysis">← Voltar</button>
    <h1>Análise da carteira</h1><p>Sinais objetivos para você decidir o que manter, vender ou reforçar</p></div>
    <div class="toolbar"><label class="field small">CDI % a.a.<input type="number" step="any" data-change="pa" data-field="cdi" value="${esc(state.settings.cdi || market.cdi || '')}" style="width:100px"></label>
      <button class="btn" data-action="update-quotes">↻ Atualizar cotações</button></div></div>
    <div class="card note">⚠️ <strong>Isto não é recomendação de investimento</strong> e não prevê o mercado: mostra fatos da sua carteira (desempenho × CDI, concentração, distância das máximas/mínimas, peso × meta e impostos) para apoiar a sua decisão.</div>`;
  if (!active.length) return head + '<div class="card empty-state"><h2>Sem posições abertas</h2></div>';

  // 1) rebalanceamento
  const byClass = {}; for (const p of active) byClass[p.asset.class] = (byClass[p.asset.class] || 0) + p.value;
  const classes = Object.keys(ASSET_CLASSES).filter(c => byClass[c] || +tg[c]);
  const sumT = sum(classes, c => +tg[c] || 0);
  const aporte = +ui.aporte || 0;
  const deficits = classes.map(c => ({ c, d: Math.max(0, ((+tg[c] || 0) / 100) * (total + aporte) - (byClass[c] || 0)) }));
  const defSum = sum(deficits, x => x.d);
  const rebal = `<div class="card"><h2>⚖️ Rebalanceamento</h2><div class="sub">defina o % alvo de cada classe (soma: <strong class="${Math.abs(sumT - 100) < 0.01 ? 'pos' : 'neg'}">${String(sumT).replace('.', ',')}%</strong>)</div>
    <div class="table-wrap"><table><thead><tr><th>Classe</th><th class="num">Atual</th><th class="num">Atual %</th><th class="num">Alvo %</th><th class="num">Para chegar no alvo</th>${aporte ? '<th class="num">No próximo aporte</th>' : ''}</tr></thead><tbody>
    ${classes.map(c => { const cur = byClass[c] || 0, t = +tg[c] || 0, diff = (t / 100) * total - cur; const ap = aporte && defSum ? (deficits.find(x => x.c === c).d / defSum) * aporte : 0;
      return `<tr><td>${ASSET_CLASSES[c].icon} ${esc(c)}</td><td class="num">${fmt(cur)}</td><td class="num">${pct(total ? cur / total : 0)}</td>
        <td class="num"><input type="number" min="0" max="100" step="1" class="mini" data-change="pa" data-field="t:${esc(c)}" value="${t || ''}"></td>
        <td class="num">${!sumT ? '—' : Math.abs(diff) < total * 0.01 ? '<span class="muted">ok</span>' : diff > 0 ? `<span class="pos">comprar ${fmt(diff)}</span>` : `<span class="neg">reduzir ${fmt(-diff)}</span>`}</td>
        ${aporte ? `<td class="num">${ap ? fmt(ap) : '—'}</td>` : ''}</tr>`; }).join('')}
    </tbody></table></div>
    <div class="toolbar mt"><label class="field small">Rebalancear só com aporte novo de (R$)<input type="number" min="0" step="100" data-change="pa-aporte" value="${aporte || ''}" style="width:140px"></label>
      <span class="small muted">rebalancear com aportes evita vender e pagar IR</span></div></div>`;

  // 2) sinais por ativo
  const rows = active.sort((a, b) => b.value - a.value).map(p => ({ p, sig: assetSignals(p, total) }));
  const nWarn = sum(rows, r => r.sig.filter(x => x.lv === 'warn').length);
  const signals = `<div class="card mt"><h2>🔎 Sinais por ativo</h2><div class="sub">${nWarn ? `${nWarn} ponto(s) de atenção` : 'nenhum ponto de atenção'} · atualize as cotações para os dados de 52 semanas</div>
    ${rows.map(({ p, sig }) => `<div class="sig-row"><div class="sig-head"><strong>${esc(p.asset.ticker)}</strong> <span class="pill">${ASSET_CLASSES[p.asset.class]?.icon || ''} ${esc(p.asset.class)}</span>
      <span class="muted small">${fmt(p.value)} · ${pct(total ? p.value / total : 0)} · <span class="${p.result >= 0 ? 'pos' : 'neg'}">${pct(p.resultPct)}</span></span>
      ${!ASSET_CLASSES[p.asset.class]?.manual ? `<button class="btn small" data-action="trade-pick" data-id="${p.asset.id}">Simular venda/troca</button>` : ''}</div>
      ${sig.length ? `<ul class="sig">${sig.map(x => `<li class="${x.lv}">${x.lv === 'warn' ? '⚠️' : x.lv === 'good' ? '✅' : 'ℹ️'} ${x.t}</li>`).join('')}</ul>` : '<p class="small muted">Sem sinais relevantes.</p>'}</div>`).join('')}</div>`;

  // 3) simulador de troca
  const tr = ui.trade || {};
  const sellable = active.filter(p => !ASSET_CLASSES[p.asset.class]?.manual);
  const sp = sellable.find(p => p.asset.id === tr.id) || sellable[0];
  let trade = '';
  if (sp) {
    const qty = Math.min(+tr.qty || sp.qty, sp.qty), price = +tr.price || sp.price;
    const gross = qty * price, gain = qty * (price - sp.avg);
    const rule = TAX[sp.asset.class] || TAX['Outros'];
    const salesMonth = monthSales(sp.asset.class) + gross;
    const exempt = rule.exempt && salesMonth <= rule.exempt;
    const tax = gain > 0 && !exempt ? gain * rule.rate : 0;
    const net = gross - tax;
    const dest = tr.dest || '';
    const after = { ...byClass }; after[sp.asset.class] = (after[sp.asset.class] || 0) - gross; if (dest) after[dest] = (after[dest] || 0) + net;
    const totAfter = total - gross + (dest ? net : 0);
    const opt = (v, l, sel) => `<option value="${esc(v)}" ${sel ? 'selected' : ''}>${esc(l)}</option>`;
    trade = `<div class="card mt" id="trade"><h2>🔁 Simular venda / troca</h2><div class="sub">veja o imposto e como fica a carteira antes de decidir</div>
      <div class="toolbar">
        <label class="field small">Vender<select data-change="tr" data-field="id">${sellable.map(p => opt(p.asset.id, `${p.asset.ticker} (${fmtNum(p.qty)})`, p === sp)).join('')}</select></label>
        <label class="field small">Quantidade<input type="number" step="any" min="0" data-change="tr" data-field="qty" value="${qty}" style="width:110px"></label>
        <label class="field small">Preço de venda<input type="number" step="any" min="0" data-change="tr" data-field="price" value="${round2(price)}" style="width:110px"></label>
        <label class="field small">Colocar o valor em<select data-change="tr" data-field="dest">${opt('', 'manter em caixa', !dest)}${Object.keys(ASSET_CLASSES).map(c => opt(c, c, dest === c)).join('')}</select></label>
      </div>
      <div class="debt-nums">
        <div><span class="label">Valor da venda</span><strong>${fmt(gross)}</strong></div>
        <div><span class="label">${gain >= 0 ? 'Lucro' : 'Prejuízo'} (vs. preço médio ${fmt(sp.avg)})</span><strong class="${gain >= 0 ? 'pos' : 'neg'}">${fmt(gain)}</strong></div>
        <div><span class="label">IR estimado</span><strong class="${tax ? 'neg' : ''}">${fmt(tax)}</strong><span class="muted small">${exempt && gain > 0 ? 'isento (limite mensal)' : gain <= 0 ? 'prejuízo: guarde para compensar' : `${pct(rule.rate)} via DARF até o fim do mês seguinte`}</span></div>
        <div><span class="label">Fica para reinvestir</span><strong>${fmt(net)}</strong></div>
      </div>
      ${rule.exempt && gain > 0 && !exempt ? `<p class="small">💡 As vendas de ${esc(sp.asset.class)} neste mês somariam ${fmt(salesMonth)}. Vendendo até ${fmt(Math.max(0, rule.exempt - monthSales(sp.asset.class)))} neste mês (e o resto no próximo), o lucro fica <strong>isento</strong>.</p>` : ''}
      <p class="small muted">${esc(rule.note)}. Prejuízos de meses anteriores podem reduzir o imposto. Confirme com a Receita ou um contador.</p>
      <h3 class="small-title">Carteira depois da troca</h3>
      <div class="table-wrap"><table><thead><tr><th>Classe</th><th class="num">Hoje</th><th class="num">Depois</th>${sumT ? '<th class="num">Alvo</th>' : ''}</tr></thead><tbody>
        ${Object.keys(ASSET_CLASSES).filter(c => byClass[c] || after[c] > 0.5).map(c => `<tr><td>${ASSET_CLASSES[c].icon} ${esc(c)}</td><td class="num">${pct(total ? (byClass[c] || 0) / total : 0)}</td><td class="num"><strong>${pct(totAfter ? Math.max(0, after[c] || 0) / totAfter : 0)}</strong></td>${sumT ? `<td class="num">${+tg[c] ? (+tg[c]) + '%' : '—'}</td>` : ''}</tr>`).join('')}
      </tbody></table></div>
      <div class="toolbar mt"><button class="btn primary" data-action="trade-register">Registrar esta venda</button><span class="small muted">cria a operação de venda na carteira (não executa nada na corretora)</span></div></div>`;
  }
  return head + `<div class="mt">${rebal}</div>` + signals + trade;
}

function viewInvestments() {
  if (ui.analysis) return viewAnalysis();
  const { list, totals } = portfolio();
  const active = list.filter(p => p.qty > 1e-9).sort((a, b) => b.value - a.value);
  const qa = state.investments.quotesAt;
  const head = `<div class="page-head"><div><h1>Investimentos</h1><p>Carteira consolidada, alocação e rentabilidade · <span class="small ${qa ? 'pos' : 'muted'}">${qa ? '⚡ cotações automáticas, atualizadas ' + new Date(qa).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' }) : 'cotações automáticas ativadas'}</span>${market.cdi ? ` · <span class="small muted">CDI ${fmtPctNum(market.cdi)}${market.ipca12 != null ? ' · IPCA 12m ' + fmtPctNum(market.ipca12) : ''}</span>` : ''}</p></div>
    <div class="toolbar"><button class="btn" data-action="open-analysis">🔎 Análise da carteira</button><button class="btn" data-action="update-quotes">↻ Atualizar cotações</button><button class="btn primary" data-action="add-op">+ Nova operação</button></div></div>`;
  if (!state.investments.operations.length) {
    return head + `<div class="card empty-state"><h2>Nenhum investimento cadastrado</h2>
      <p>Registre compras, vendas e proventos, ou importe o CSV de negociações da Área do Investidor da B3 / da sua corretora. As cotações de ações, FIIs, ETFs, BDRs e cripto são atualizadas automaticamente.</p>
      <button class="btn primary" data-action="add-op">+ Nova operação</button> <button class="btn" data-action="nav" data-view="import">Importar CSV</button></div>`;
  }
  const kpis = `<div class="grid kpis">
    <div class="card kpi"><div class="label">Patrimônio investido</div><div class="value">${fmt(totals.value)}</div><div class="hint">${active.length} ativos</div></div>
    <div class="card kpi"><div class="label">Total aplicado (custo)</div><div class="value">${fmt(totals.cost)}</div></div>
    <div class="card kpi"><div class="label">Resultado não realizado</div><div class="value ${totals.result >= 0 ? 'pos' : 'neg'}">${fmt(totals.result)}</div><div class="hint">${pct(totals.resultPct)} sobre o custo</div></div>
    <div class="card kpi"><div class="label">Proventos recebidos</div><div class="value pos">${fmt(totals.dividends)}</div><div class="hint">lucro realizado em vendas: ${fmt(totals.realized)}</div></div>
  </div>`;

  const byClass = {};
  for (const p of active) byClass[p.asset.class] = (byClass[p.asset.class] || 0) + p.value;
  const clsItems = Object.entries(byClass).map(([c, v]) => ({ tag: c, label: c, value: v, color: ASSET_CLASSES[c]?.color || '#a3a3a3', icon: ASSET_CLASSES[c]?.icon })).sort((a, b) => b.value - a.value);
  const alloc = `<div class="card"><h2>Alocação por classe</h2><div class="sub">distribuição do patrimônio</div>
    <div class="donut-wrap">${donutSVG(clsItems, totals.value, fmt(totals.value), 'patrimônio')}${legendHTML(clsItems, totals.value)}</div></div>`;

  const months = Array.from({ length: 12 }, (_, i) => addMonths(currentMonth(), i - 11));
  const hist = state.investments.history || {};
  const histMonths = Object.keys(hist).sort().slice(-12);
  const evo = `<div class="card"><h2>Aportes por mês</h2><div class="sub">compras − vendas, últimos 12 meses</div>
    ${barsSVG(months.map(monthShort), [{ label: 'Aportes', color: 'var(--blue)', values: aportesByMonth(months) }])}
    ${histMonths.length >= 2 ? `<h2 class="mt">Evolução do patrimônio</h2>${lineSVG(histMonths.map(monthShort), histMonths.map(m => hist[m]))}` : '<p class="muted small">A evolução do patrimônio aparece aqui conforme você atualiza as cotações a cada mês.</p>'}
  </div>`;

  const rows = active.map(p => {
    const a = p.asset; const manual = ASSET_CLASSES[a.class]?.manual;
    const upd = a.priceUpdatedAt ? `atualizado ${new Date(a.priceUpdatedAt).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' })}` : 'preço manual';
    const idxLabel = p.market ? `<br><span class="small pos">mercado · PU ${fmt(a.tdPrice)}</span>` : p.auto ? `<br><span class="small pos">auto · ${a.indexer === 'cdi' ? a.idxRate + '% CDI' : a.indexer === 'pre' ? 'pré ' + fmtRate(a.idxRate) : 'IPCA+' + fmtRate(a.idxRate)}</span>` : (manual ? '<br><span class="small warn">manual</span>' : '');
    return `<tr>
      <td><strong>${esc(a.ticker)}</strong></td>
      <td><span class="pill">${ASSET_CLASSES[a.class]?.icon || ''} ${esc(a.class)}</span></td>
      <td class="num">${manual ? '—' : fmtNum(p.qty)}</td>
      <td class="num">${manual ? '—' : fmt(p.avg)}</td>
      <td class="num" title="${esc(upd)}">${manual ? '—' : fmt(p.price)}</td>
      <td class="num">${fmt(p.cost)}</td>
      <td class="num"><strong>${fmt(p.value)}</strong>${idxLabel}</td>
      <td class="num ${p.result >= 0 ? 'pos' : 'neg'}">${fmt(p.result)}<br><span class="small">${pct(p.resultPct)}</span></td>
      <td class="num">${pct(totals.value ? p.value / totals.value : 0)}</td>
      <td><div class="row-actions">
        <button class="icon-btn" title="Nova operação" data-action="add-op" data-ticker="${esc(a.ticker)}">＋</button>
        <button class="icon-btn" title="${manual ? 'Atualizar valor atual' : 'Editar preço/ativo'}" data-action="edit-asset" data-id="${a.id}">✎</button>
      </div></td></tr>`;
  }).join('');
  const table = `<div class="card mt"><h2>Ativos</h2><div class="sub">passe o mouse no preço para ver quando foi atualizado · renda fixa e Tesouro: atualize o valor atual com ✎</div>
    <div class="table-wrap"><table><thead><tr><th>Ativo</th><th>Classe</th><th class="num">Qtd</th><th class="num">Preço médio</th><th class="num">Preço atual</th><th class="num">Custo</th><th class="num">Valor atual</th><th class="num">Resultado</th><th class="num">% carteira</th><th></th></tr></thead>
    <tbody>${rows || '<tr><td colspan="10" class="muted">Nenhuma posição aberta.</td></tr>'}</tbody></table></div></div>`;

  const ops = [...state.investments.operations].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 30);
  const assetById = id => state.investments.assets.find(a => a.id === id);
  const opsTable = `<div class="card mt"><h2>Últimas operações</h2>
    <div class="table-wrap"><table><thead><tr><th>Data</th><th>Ativo</th><th>Tipo</th><th class="num">Qtd</th><th class="num">Preço</th><th class="num">Total</th><th></th></tr></thead><tbody>
    ${ops.map(o => `<tr><td>${fmtDate(o.date)}</td><td>${esc(assetById(o.assetId)?.ticker || '?')}</td><td>${o.type}</td><td class="num">${fmtNum(o.qty)}</td><td class="num">${fmt(o.price)}</td>
      <td class="num ${o.type === 'venda' || o.type === 'provento' ? 'pos' : ''}">${fmt(o.qty * o.price + (o.type === 'compra' ? (+o.fees || 0) : -(+o.fees || 0)))}</td>
      <td><button class="icon-btn" title="Excluir" data-action="del-op" data-id="${o.id}">✕</button></td></tr>`).join('')}
    </tbody></table></div></div>`;

  return head + kpis + `<div class="grid two">${alloc}${evo}</div>` + table + opsTable;
}

/* ===================================================== financiamentos e consórcios */
const SIM = {}; // estado dos simuladores (só na tela, não é salvo)
function isPaidMonth(ym, day) {
  const cur = currentMonth();
  if (ym < cur) return true;
  if (ym > cur) return false;
  return +todayISO().slice(8) >= Math.min(day, daysInMonth(ym));
}
const pmt = (pv, i, n) => (i ? (pv * i) / (1 - Math.pow(1 + i, -n)) : pv / n);
const dueDate = (ym, day) => `${ym}-${pad(Math.min(day, daysInMonth(ym)))}`;
const fmtRate = v => String(+v || 0).replace('.', ',') + '%';
function monthsText(k) {
  const y = Math.floor(k / 12), m = k % 12;
  return [y ? `${y} ano${y > 1 ? 's' : ''}` : '', m ? `${m} ${m > 1 ? 'meses' : 'mês'}` : ''].filter(Boolean).join(' e ') || '0 meses';
}

/** Tabela de amortização (SAC ou Price), com amortizações extras e ajuste pelo saldo informado pelo banco. */
function financingSchedule(d, more = []) {
  const i = Math.pow(1 + (+d.rate || 0) / 100, 1 / 12) - 1;
  const n = +d.term, fees = +d.fees || 0, start = d.firstDue.slice(0, 7), price = d.system === 'PRICE';
  const extras = [...(d.extras || []), ...more].map(e => ({ ...e, ym: e.date.slice(0, 7) })).sort((a, b) => a.ym.localeCompare(b.ym));
  let saldo = +d.principal, inst = price ? pmt(saldo, i, n) : saldo / n, ei = 0, nEff = n;
  const rows = []; const applied = {}; rows.applied = applied;
  const has = v => v !== undefined && v !== null && v !== '';
  const recalc = m => { const left = Math.max(1, nEff - m); inst = price ? pmt(saldo, i, left) : saldo / left; };
  // Aplica o que o banco informou: saldo devedor, parcela e/ou parcelas restantes.
  const applyBank = (bk, m, recalcIfOnlyBalance) => {
    if (has(bk.balance)) saldo = +bk.balance;
    if (+bk.remaining > 0) nEff = m + +bk.remaining;
    if (+bk.installment > 0) inst = price ? +bk.installment - fees : Math.max(0, +bk.installment - fees - saldo * i);
    else if (+bk.remaining > 0 || recalcIfOnlyBalance) recalc(m);
  };
  for (let m = 0; m < n + 600 && saldo > 0.005; m++) {
    const ym = addMonths(start, m);
    if (d.anchor && d.anchor.ym === ym) applyBank(d.anchor, m, true);
    while (ei < extras.length && extras[ei].ym <= ym) {
      const e = extras[ei++];
      const v = Math.min(saldo, +e.value || 0);
      saldo -= v; if (v > 0) applied[ym] = (applied[ym] || 0) + v;
      if (e.mode === 'parcela') recalc(m);
      if (has(e.bankBalance) || +e.bankInstallment > 0 || +e.bankRemaining > 0) applyBank({ balance: e.bankBalance, installment: e.bankInstallment, remaining: e.bankRemaining }, m, e.mode === 'parcela');
    }
    if (saldo <= 0.005) break;
    const juros = saldo * i;
    const amort = Math.min(price ? inst - juros : inst, saldo);
    saldo -= amort;
    rows.push({ ym, n: m + 1, juros, amort, fees, total: amort + juros + fees, saldo });
  }
  return rows;
}
function financingStatus(d, rows = financingSchedule(d)) {
  const day = +d.firstDue.slice(8, 10);
  const paid = rows.filter(r => isPaidMonth(r.ym, day)).length;
  const unpaid = rows.slice(paid);
  const saldo = unpaid.length ? unpaid[0].saldo + unpaid[0].amort : 0;
  return {
    rows, day, paid, total: rows.length, saldo, next: unpaid[0],
    remainingInterest: sum(unpaid, r => r.juros), remainingTotal: sum(unpaid, r => r.total),
    paidTotal: sum(rows.slice(0, paid), r => r.total), endYm: rows.length ? rows[rows.length - 1].ym : null,
    pct: +d.principal ? Math.max(0, 1 - saldo / +d.principal) : 1,
  };
}
function realizedSavings(d) {
  if (!(d.extras || []).length) return null;
  const pure = financingSchedule({ ...d, extras: [], anchor: null });
  const real = financingSchedule({ ...d, anchor: null });
  return { interest: sum(pure, r => r.juros) - sum(real, r => r.juros), months: pure.length - real.length, total: sum(d.extras, e => +e.value || 0) };
}
function extraForm(e = {}) {
  const v = k => (e[k] === undefined || e[k] === null ? '' : e[k]);
  return `<div class="form-grid">
    <label>Data em que pagou<input type="date" name="date" required value="${e.date || todayISO()}"></label>
    <label>Valor amortizado (R$)<input type="number" step="0.01" min="0" name="value" required value="${v('value')}"></label>
    <label class="full">O banco aplicou para<select name="mode"><option value="prazo">Reduzir o prazo</option><option value="parcela" ${e.mode === 'parcela' ? 'selected' : ''}>Reduzir a parcela</option></select></label>
    <div class="field full"><strong class="small" style="color:var(--text)">Depois da amortização, o banco mostrou (opcional — deixa a tabela exata):</strong></div>
    <label>Saldo devedor (R$)<input type="number" step="0.01" min="0" name="bankBalance" value="${v('bankBalance')}"></label>
    <label>Nova parcela (R$)<input type="number" step="0.01" min="0" name="bankInstallment" value="${v('bankInstallment')}"></label>
    <label>Parcelas restantes<input type="number" step="1" min="0" name="bankRemaining" value="${v('bankRemaining')}"></label>
  </div><p class="muted small">Você pode registrar agora e voltar depois (✎) para preencher os números quando o banco atualizar.</p>`;
}
function extraFromForm(f) {
  const n = k => (f[k] === '' || f[k] === undefined ? '' : +f[k]);
  if (!(+f.value > 0)) { toast('Informe o valor.'); return null; }
  return { date: f.date, value: +f.value, mode: f.mode, bankBalance: n('bankBalance'), bankInstallment: n('bankInstallment'), bankRemaining: n('bankRemaining') };
}
function simulateAmort(d, value, mode) {
  const base = financingStatus(d);
  if (!base.next || !(value > 0)) return null;
  const sim = financingStatus(d, financingSchedule(d, [{ date: base.next.ym + '-01', value, mode }]));
  return { interestSaved: base.remainingInterest - sim.remainingInterest, monthsLess: base.total - sim.total, oldInstallment: base.next.total, newInstallment: sim.next ? sim.next.total : 0, newEnd: sim.endYm };
}

function consorcioSchedule(c) {
  const n = +c.term, credit = +c.credit;
  const base = (credit * (1 + (+c.adminFee || 0) / 100 + (+c.reserveFund || 0) / 100)) / n + (credit * (+c.insurance || 0)) / 100;
  const adj = (+c.adjust || 0) / 100, start = c.firstDue.slice(0, 7);
  return Array.from({ length: n }, (_, m) => ({ ym: addMonths(start, m), n: m + 1, total: base * Math.pow(1 + adj, Math.floor(m / 12)) }));
}
function consorcioStatus(c) {
  const rows = consorcioSchedule(c), day = +c.firstDue.slice(8, 10);
  const override = c.paidOverride !== '' && c.paidOverride != null;
  const paid = override ? Math.min(+c.paidOverride, rows.length) : rows.filter(r => isPaidMonth(r.ym, day)).length;
  const bid = c.contemplated ? +c.bid || 0 : 0;
  const unpaid = rows.slice(paid);
  return {
    rows, day, paid, total: rows.length, next: unpaid[0],
    paidTotal: sum(rows.slice(0, paid), r => r.total) + bid,
    remaining: Math.max(0, sum(unpaid, r => r.total) - bid),
    endYm: rows.length ? rows[rows.length - 1].ym : null,
    pct: rows.length ? paid / rows.length : 0,
    creditNow: +c.credit * Math.pow(1 + (+c.adjust || 0) / 100, Math.floor(Math.max(0, paid - 1) / 12)),
  };
}
const debtStatus = d => (d.kind === 'consorcio' ? consorcioStatus(d) : financingStatus(d));
function debtPaymentFor(id, ym) {
  const d = state.debts.find(x => x.id === id); if (!d) return 0;
  const rows = d.kind === 'consorcio' ? consorcioSchedule(d) : financingSchedule(d);
  return round2(rows.find(r => r.ym === ym)?.total || 0);
}
const debtAvatar = d => avatar({ name: d.institution || d.name, type: 'corrente' });

function ensureDebtTag(kind) {
  const def = kind === 'consorcio'
    ? { name: 'Consórcio', color: '#0d9488', icon: '🤝', group: 'essencial', exclude: false, rule: 'consorcio|consorc' }
    : { name: 'Financiamento', color: '#4f46e5', icon: '🏡', group: 'essencial', exclude: false, rule: 'habitacao|habitacional|financiamento imob|prestacao imovel|prestacao habit' };
  if (!state.tags.some(t => t.name === def.name)) {
    const { rule, ...tag } = def; state.tags.push(tag);
    state.rules.unshift({ id: uid(), pattern: rule, tag: def.name, sign: 'out' });
  }
  return def.name;
}
/** Cria/atualiza/remove o lançamento recorrente da parcela (valor calculado mês a mês). */
function syncDebtRecurring(d, on, accountId) {
  const idx = state.recurring.findIndex(r => r.debtId === d.id);
  if (!on) { if (idx >= 0) state.recurring.splice(idx, 1); return; }
  const st = debtStatus(d);
  const base = idx >= 0 ? state.recurring[idx] : { id: uid(), generated: [], startMonth: st.next?.ym || currentMonth() };
  Object.assign(base, { debtId: d.id, description: `Parcela ${d.name}`, amount: -round2(st.next?.total || 0), day: st.day, tag: ensureDebtTag(d.kind), accountId });
  if (idx < 0) state.recurring.push(base);
}

function simText(d) {
  const st = SIM[d.id];
  if (!st || !(+st.value > 0)) return '<span class="muted">Digite um valor para ver quanto economiza de juros.</span>';
  const r = simulateAmort(d, +st.value, st.mode || 'prazo');
  if (!r) return '';
  return (st.mode || 'prazo') === 'parcela'
    ? `A parcela cai de <strong>${fmt(r.oldInstallment)}</strong> para <strong class="pos">${fmt(r.newInstallment)}</strong> e você economiza <strong class="pos">${fmt(r.interestSaved)}</strong> em juros.`
    : `Você quita <strong class="pos">${monthsText(r.monthsLess)} antes</strong> (${r.newEnd ? monthLabel(r.newEnd) : 'agora'}) e economiza <strong class="pos">${fmt(r.interestSaved)}</strong> em juros.`;
}
function consSimText(c) {
  const st = SIM[c.id] || {}; const s = consorcioStatus(c);
  const bid = +st.bid || 0, emb = Math.min(+st.emb || 0, bid);
  if (!bid) return '<span class="muted">Informe o % de lance para ver os valores.</span>';
  const bidV = (s.creditNow * bid) / 100, embV = (s.creditNow * emb) / 100, own = bidV - embV;
  const parc = s.next ? own / s.next.total : 0;
  return `Lance de ${fmtRate(bid)} = <strong>${fmt(bidV)}</strong>${emb ? ` (${fmt(embV)} embutido na carta + <strong>${fmt(own)}</strong> do seu bolso)` : ''}. Carta liberada: <strong>${fmt(s.creditNow - embV)}</strong>.${parc ? ` O valor do seu bolso equivale a ~${parc.toFixed(1).replace('.', ',')} parcelas atuais.` : ''}`;
}

function debtProgress(s, extra = '') {
  return `<div class="goal"><div class="goal-top"><span class="small"><strong>${s.paid}</strong> de ${s.total} parcelas pagas</span><span class="small">${pct(s.pct)} ${extra}</span></div>
    <div class="bar"><span style="width:${Math.min(100, s.pct * 100)}%;background:var(--green)"></span></div></div>`;
}

function financingCard(d) {
  const s = financingStatus(d);
  const left = s.total - s.paid;
  const pts = s.rows.filter((r, i) => i % 12 === 0 || i === s.rows.length - 1);
  const chart = pts.length > 2 ? lineSVG(pts.map(r => r.ym.slice(2, 4)), [+d.principal, ...pts.map(r => r.saldo)].slice(1)) : '';
  const sv = realizedSavings(d);
  const extras = (d.extras || []).length ? `<h3 class="mt small-title">Amortizações realizadas</h3>
    ${sv && sv.interest > 1 ? `<p class="small">✅ Você já amortizou <strong>${fmt(sv.total)}</strong> e isso economiza cerca de <strong class="pos">${fmt(sv.interest)}</strong> em juros${sv.months > 0 ? ` e <strong class="pos">${monthsText(sv.months)}</strong> de prazo` : ''}.</p>` : ''}
    ${d.extras.map((e, i) => { const bank = [e.bankBalance !== '' && e.bankBalance != null ? 'saldo ' + fmt(e.bankBalance) : '', +e.bankInstallment ? 'parcela ' + fmt(e.bankInstallment) : '', +e.bankRemaining ? e.bankRemaining + ' parcelas restantes' : ''].filter(Boolean).join(' · ');
      return `<div class="list-row small"><span class="ticon sm" style="--c:var(--green)">✔</span><span class="grow">${fmtDate(e.date)} · <strong>${fmt(e.value)}</strong> · ${e.mode === 'parcela' ? 'reduziu a parcela' : 'reduziu o prazo'}
        <br><span class="${bank ? 'muted' : 'warn'}">${bank ? 'banco: ' + bank : '⚠ falta informar os números do banco'}</span></span>
        <button class="icon-btn" title="Atualizar com os dados do banco" data-action="edit-extra" data-id="${d.id}" data-idx="${i}">✎</button><button class="icon-btn" title="Excluir" data-action="del-extra" data-id="${d.id}" data-idx="${i}">✕</button></div>`; }).join('')}` : '';
  const next12 = s.rows.slice(s.paid, s.paid + 12);
  return `<div class="card debt">
    <div class="card-head"><div class="debt-title">${debtAvatar(d)}<div><h2>${esc(d.name)}</h2>
      <div class="sub">${esc(d.institution || '')} · ${d.system === 'PRICE' ? 'Price' : 'SAC'} · ${fmtRate(d.rate)} a.a. · ${d.term} meses${+d.assetValue ? ` · bem ${fmt(d.assetValue)}` : ''}</div></div></div>
      <div class="row-actions"><button class="icon-btn" title="Editar" data-action="edit-debt" data-id="${d.id}">✎</button><button class="icon-btn" title="Excluir" data-action="del-debt" data-id="${d.id}">✕</button></div></div>
    ${debtProgress(s, 'quitado')}
    <div class="debt-nums">
      <div><span class="label">Saldo devedor</span><strong>${fmt(s.saldo)}</strong></div>
      <div><span class="label">Próxima parcela</span><strong>${s.next ? fmt(s.next.total) : '—'}</strong><span class="muted small">${s.next ? fmtDate(dueDate(s.next.ym, s.day)) : 'quitado 🎉'}</span></div>
      <div><span class="label">Juros a pagar</span><strong class="neg">${fmt(s.remainingInterest)}</strong></div>
      <div><span class="label">Quitação prevista</span><strong>${s.endYm ? monthLabel(s.endYm) : '—'}</strong><span class="muted small">faltam ${monthsText(left)}</span></div>
    </div>
    ${chart ? `<div class="small muted">Saldo devedor ao longo do tempo (ano)</div>${chart}` : ''}
    ${s.next ? `<div class="sim"><h3>💡 Simular amortização extra</h3>
      <div class="toolbar"><input type="number" min="0" step="100" placeholder="Valor (R$)" data-input="sim-fin" data-id="${d.id}" value="${esc(SIM[d.id]?.value || '')}">
        <select data-change="sim-fin-mode" data-id="${d.id}"><option value="prazo">Reduzir prazo</option><option value="parcela" ${SIM[d.id]?.mode === 'parcela' ? 'selected' : ''}>Reduzir parcela</option></select></div>
      <div id="sim-${d.id}" class="sim-out">${simText(d)}</div></div>` : ''}
    ${(() => { if (!d.plan?.length) return ''; const r = planResult(d, d.plan); return r ? `<p class="small">📊 Plano salvo: quita em <strong>${monthLabel(r.P.end)}</strong> e economiza <strong class="pos">${fmt(r.saved)}</strong>.</p>` : ''; })()}
    <div class="toolbar mt"><button class="btn small primary" data-action="open-sim" data-id="${d.id}">📊 Simulador completo</button>
      <button class="btn small" data-action="add-extra" data-id="${d.id}">+ Registrar amortização feita</button>
      <button class="btn small" data-action="anchor-debt" data-id="${d.id}">🏦 Atualizar pelos dados do banco</button></div>
    ${extras}
    ${d.anchor ? `<p class="small muted">Ajustado pelo banco antes da parcela de ${monthLabel(d.anchor.ym)}: saldo ${fmt(d.anchor.balance)}${+d.anchor.installment ? ' · parcela ' + fmt(d.anchor.installment) : ''}${+d.anchor.remaining ? ' · ' + d.anchor.remaining + ' parcelas restantes' : ''}.</p>` : ''}
    <details><summary>Próximas parcelas</summary><div class="table-wrap"><table><thead><tr><th>Nº</th><th>Vencimento</th><th class="num">Amortização</th><th class="num">Juros</th><th class="num">Seguros/taxas</th><th class="num">Parcela</th><th class="num">Saldo após</th></tr></thead>
      <tbody>${next12.map(r => `<tr><td>${r.n}</td><td>${fmtDate(dueDate(r.ym, s.day))}</td><td class="num">${fmt(r.amort)}</td><td class="num">${fmt(r.juros)}</td><td class="num">${fmt(r.fees)}</td><td class="num"><strong>${fmt(r.total)}</strong></td><td class="num">${fmt(r.saldo)}</td></tr>`).join('')}</tbody></table></div></details>
  </div>`;
}

function consorcioCard(c) {
  const s = consorcioStatus(c);
  return `<div class="card debt">
    <div class="card-head"><div class="debt-title">${debtAvatar(c)}<div><h2>${esc(c.name)}</h2>
      <div class="sub">${esc(c.institution || '')} · carta ${fmt(s.creditNow)} · ${c.term} meses</div></div></div>
      <div class="row-actions"><button class="icon-btn" title="Editar" data-action="edit-debt" data-id="${c.id}">✎</button><button class="icon-btn" title="Excluir" data-action="del-debt" data-id="${c.id}">✕</button></div></div>
    <p>${c.contemplated ? `<span class="pill badge-ok">✓ Contemplado${c.contemplatedDate ? ' em ' + fmtDate(c.contemplatedDate) : ''}</span>${+c.bid ? ` <span class="small muted">lance de ${fmt(c.bid)}</span>` : ''}` : '<span class="pill">⏳ Aguardando contemplação</span>'}</p>
    ${debtProgress(s, 'pago')}
    <div class="debt-nums">
      <div><span class="label">Parcela atual</span><strong>${s.next ? fmt(s.next.total) : '—'}</strong><span class="muted small">${s.next ? 'vence ' + fmtDate(dueDate(s.next.ym, s.day)) : 'encerrado 🎉'}</span></div>
      <div><span class="label">Total pago</span><strong>${fmt(s.paidTotal)}</strong></div>
      <div><span class="label">Falta pagar</span><strong class="neg">${fmt(s.remaining)}</strong></div>
      <div><span class="label">Encerramento</span><strong>${s.endYm ? monthLabel(s.endYm) : '—'}</strong><span class="muted small">faltam ${monthsText(s.total - s.paid)}</span></div>
    </div>
    <p class="small muted">Taxa de administração ${fmtRate(c.adminFee)} · fundo de reserva ${fmtRate(c.reserveFund)}${+c.insurance ? ` · seguro ${fmtRate(c.insurance)} a.m.` : ''}${+c.adjust ? ` · reajuste estimado ${fmtRate(c.adjust)} a.a.` : ''}</p>
    ${!c.contemplated ? `<div class="toolbar"><button class="btn small primary" data-action="open-cons" data-id="${c.id}">🎯 Estratégia de contemplação</button></div>` : ''}
    ${!c.contemplated && s.next ? `<div class="sim"><h3>🎯 Simular lance</h3>
      <div class="toolbar"><label class="field">Lance (% da carta)<input type="number" min="0" max="100" step="1" data-input="sim-cons" data-field="bid" data-id="${c.id}" value="${esc(SIM[c.id]?.bid || '')}"></label>
        <label class="field">Lance embutido (%)<input type="number" min="0" max="30" step="1" data-input="sim-cons" data-field="emb" data-id="${c.id}" value="${esc(SIM[c.id]?.emb || '')}"></label></div>
      <div id="csim-${c.id}" class="sim-out">${consSimText(c)}</div>
      <p class="small muted">Calculado sobre a carta atualizada; algumas administradoras calculam sobre o saldo devedor.</p></div>` : ''}
  </div>`;
}

/* ---------- simulador completo de amortização ---------- */
const FREQS = { 0: 'uma vez', 1: 'todo mês', 6: 'a cada 6 meses', 12: 'todo ano', 24: 'a cada 2 anos' };
function expandPlan(items, fromYm, untilYm) {
  const out = [];
  for (const it of items) {
    const v = +it.value, f = +it.every || 0;
    if (!(v > 0)) continue;
    let ym = it.start || fromYm;
    if (f) while (ym < fromYm) ym = addMonths(ym, f);       // recorrente que começou antes: alinha no próximo
    else if (ym < fromYm) ym = fromYm;
    const end = it.end && it.end < untilYm ? it.end : untilYm;
    for (let k = 0; ym <= end && k < 1200; k++) { out.push({ date: ym + '-01', value: v, mode: it.mode || 'prazo' }); if (!f) break; ym = addMonths(ym, f); }
  }
  return out;
}
function planResult(d, items) {
  const base = financingStatus(d);
  if (!base.next) return null;
  const from = base.next.ym;
  const plan = financingStatus(d, financingSchedule(d, expandPlan(items, from, base.endYm)));
  const fut = st => {
    const extra = sum(Object.entries(st.rows.applied).filter(([ym]) => ym >= from), ([, v]) => v);
    return { st, months: st.total - st.paid, interest: st.remainingInterest, parcels: st.remainingTotal, extra, out: st.remainingTotal + extra, end: st.endYm, next: st.next ? st.next.total : 0 };
  };
  const A = fut(base), P = fut(plan);
  return { from, A, P, saved: A.out - P.out, monthsLess: A.months - P.months };
}

function lineMultiSVG(labels, series, { height = 220 } = {}) {
  const W = 640, H = height, L = 56, B = 24, T = 10;
  const all = series.flatMap(s => s.values).filter(v => v != null);
  const max = Math.max(1, ...all), min = Math.min(0, ...all), range = max - min || 1;
  const x = i => L + ((W - L - 12) * i) / Math.max(1, labels.length - 1);
  const y = v => T + (H - T - B) * (1 - (v - min) / range);
  let g = '';
  for (let k = 0; k <= 4; k++) { const v = min + (range * k) / 4; g += `<line class="grid-line" x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtShort(v).replace('R$ ', '')}</text>`; }
  const step = Math.ceil(labels.length / 12);
  labels.forEach((l, i) => { if (i % step === 0 || i === labels.length - 1) g += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(l)}</text>`; });
  for (const s of series) {
    const pts = s.values.map((v, i) => (v == null ? null : `${x(i)},${y(v)}`)).filter(Boolean);
    g += `<polyline points="${pts.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="6 5"' : ''}/>`;
    s.values.forEach((v, i) => { if (v != null) g += `<circle cx="${x(i)}" cy="${y(v)}" r="6" fill="transparent"><title>${esc(s.label)} — ${esc(labels[i])}: ${fmt(v)}</title></circle>`; });
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${g}</svg><div class="chart-legend">${series.map(s => `<span><span class="dot" style="background:${s.color}"></span> ${esc(s.label)}</span>`).join('')}</div>`;
}

function viewSimulator() {
  const d = state.debts.find(x => x.id === ui.sim.debtId);
  if (!d) { ui.sim = null; return viewDebts(); }
  const items = ui.sim.items;
  const r = planResult(d, items);
  const head = `<div class="page-head"><div><button class="btn small ghost" data-action="close-sim">← Voltar</button>
    <h1>Simulador de amortização</h1><p>${esc(d.name)} · ${esc(d.institution || '')} · ${d.system === 'PRICE' ? 'Price' : 'SAC'} · ${fmtRate(d.rate)} a.a.</p></div></div>`;
  if (!r) return head + '<div class="card empty-state"><h2>Financiamento quitado 🎉</h2></div>';
  const { from, A, P } = r;
  const nextLabel = monthLabel(from);

  const hero = items.length
    ? `<div class="card sim-hero"><div class="sim-hero-icon">🎉</div><div>
        <div class="big-number pos">Economia de ${fmt(r.saved)}</div>
        <div>Com esse plano você quita em <strong>${monthLabel(P.end)}</strong>${r.monthsLess > 0 ? ` — <strong>${monthsText(r.monthsLess)} antes</strong> do previsto (${monthLabel(A.end)})` : ''}.
        ${P.next && Math.abs(P.next - A.next) > 0.5 ? ` A próxima parcela fica em <strong>${fmt(P.next)}</strong> (hoje ${fmt(A.next)}).` : ''}</div></div></div>`
    : `<div class="card sim-hero"><div class="sim-hero-icon">💡</div><div><strong>Monte seu plano abaixo</strong><div class="muted">Adicione amortizações únicas ou recorrentes (todo mês, 13º, FGTS a cada 2 anos…) e veja na hora quando quita e quanto economiza.</div></div></div>`;

  const freqOpts = Object.entries(FREQS).map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
  const builder = `<div class="card"><h2>Seu plano de amortizações</h2><div class="sub">próxima parcela em ${nextLabel} · as amortizações entram antes da parcela do mês</div>
    <div class="toolbar"><span class="small muted">Atalhos:</span>
      <button class="btn small" data-action="sim-preset" data-value="500" data-every="1" data-start="${from}">R$ 500 todo mês</button>
      <button class="btn small" data-action="sim-preset" data-value="3000" data-every="12" data-start="${from.slice(0, 4)}-12">13º salário (dezembro)</button>
      <button class="btn small" data-action="sim-preset" data-value="15000" data-every="24" data-start="${from}">FGTS a cada 2 anos</button>
      <button class="btn small" data-action="sim-preset" data-value="20000" data-every="0" data-start="${from}">Valor único</button></div>
    <div class="form-grid sim-form">
      <label>Valor (R$)<input type="number" id="sp-value" min="0" step="100" placeholder="1000"></label>
      <label>Frequência<select id="sp-every">${freqOpts}</select></label>
      <label>A partir de<input type="month" id="sp-start" value="${from}" min="${from}"></label>
      <label>Até (opcional)<input type="month" id="sp-end"></label>
      <label class="full">Efeito<select id="sp-mode"><option value="prazo">Reduzir o prazo (economiza mais juros)</option><option value="parcela">Reduzir o valor da parcela</option></select></label>
    </div>
    <div class="toolbar mt"><button class="btn primary" data-action="sim-add">+ Adicionar ao plano</button></div>
    ${items.length ? items.map((it, i) => `<div class="list-row"><span class="ticon sm" style="--c:var(--green)">${+it.every ? '🔁' : '💵'}</span>
      <span class="grow"><strong>${fmt(it.value)}</strong> · ${FREQS[+it.every || 0]} · ${+it.every ? 'desde' : 'em'} ${monthShort(it.start)}${it.end ? ' até ' + monthShort(it.end) : ''}
      <br><span class="muted small">${it.mode === 'parcela' ? 'reduz a parcela' : 'reduz o prazo'}</span></span>
      <button class="btn small soft" data-action="sim-realize" data-idx="${i}" title="Registrar como realizada">✔ Realizei</button>
      <button class="icon-btn" data-action="sim-del" data-idx="${i}">✕</button></div>`).join('') : '<p class="muted small">Nenhuma amortização no plano ainda.</p>'}
    <div class="toolbar mt"><button class="btn" data-action="sim-save">💾 Salvar plano</button>${items.length ? '<button class="btn ghost" data-action="sim-clear">Limpar</button>' : ''}
      ${JSON.stringify(d.plan || []) !== JSON.stringify(items) ? '<span class="small muted">alterações não salvas</span>' : (items.length ? '<span class="small pos">plano salvo ✓</span>' : '')}</div>
  </div>`;

  const diff = (a, b, invert) => { const v = b - a; if (Math.abs(v) < 0.5) return '<span class="muted">—</span>'; const good = invert ? v > 0 : v < 0; return `<span class="${good ? 'pos' : 'neg'}">${v > 0 ? '+' : '−'}${fmt(Math.abs(v))}</span>`; };
  const compare = `<div class="card"><h2>Hoje × com o plano</h2><div class="sub">a partir de ${nextLabel}</div>
    <div class="table-wrap"><table class="compare"><thead><tr><th></th><th class="num">Sem amortizar</th><th class="num">Com o plano</th><th class="num">Diferença</th></tr></thead><tbody>
      <tr><td>Quitação</td><td class="num">${monthLabel(A.end)}</td><td class="num"><strong>${monthLabel(P.end)}</strong></td><td class="num">${r.monthsLess > 0 ? `<span class="pos">${monthsText(r.monthsLess)} antes</span>` : '<span class="muted">—</span>'}</td></tr>
      <tr><td>Parcelas restantes</td><td class="num">${A.months}</td><td class="num"><strong>${P.months}</strong></td><td class="num">${A.months - P.months ? `<span class="pos">−${A.months - P.months}</span>` : '<span class="muted">—</span>'}</td></tr>
      <tr><td>Próxima parcela</td><td class="num">${fmt(A.next)}</td><td class="num"><strong>${fmt(P.next)}</strong></td><td class="num">${diff(A.next, P.next)}</td></tr>
      <tr><td>Juros a pagar</td><td class="num">${fmt(A.interest)}</td><td class="num"><strong>${fmt(P.interest)}</strong></td><td class="num">${diff(A.interest, P.interest)}</td></tr>
      <tr><td>Parcelas (soma)</td><td class="num">${fmt(A.parcels)}</td><td class="num">${fmt(P.parcels)}</td><td class="num">${diff(A.parcels, P.parcels)}</td></tr>
      <tr><td>Amortizações extras</td><td class="num">—</td><td class="num">${fmt(P.extra)}</td><td class="num"></td></tr>
      <tr class="total"><td><strong>Total a desembolsar</strong></td><td class="num"><strong>${fmt(A.out)}</strong></td><td class="num"><strong>${fmt(P.out)}</strong></td><td class="num"><strong>${diff(A.out, P.out)}</strong></td></tr>
    </tbody></table></div></div>`;

  // séries anuais para os gráficos
  const ks = []; for (let k = 0; k < A.months; k += 12) ks.push(k); ks.push(A.months - 1);
  const rowAt = (st, ym) => st.rows.find(x => x.ym === ym);
  const labels = ks.map(k => addMonths(from, k).slice(0, 4));
  const val = (st, end, k, f) => { const ym = addMonths(from, k); const row = rowAt(st, ym); return row ? f(row) : (ym > end ? (f === saldoF ? 0 : null) : null); };
  const saldoF = x => x.saldo, parcF = x => x.total;
  const charts = `<div class="grid two">
    <div class="card"><h2>Saldo devedor</h2><div class="sub">no início de cada ano</div>${lineMultiSVG(labels, [
      { label: 'Sem amortizar', color: 'var(--muted)', dash: true, values: ks.map(k => val(A.st, A.end, k, saldoF)) },
      { label: 'Com o plano', color: 'var(--green)', values: ks.map(k => val(P.st, P.end, k, saldoF)) }])}</div>
    <div class="card"><h2>Valor da parcela</h2><div class="sub">parcela + seguros/taxas</div>${lineMultiSVG(labels, [
      { label: 'Sem amortizar', color: 'var(--muted)', dash: true, values: ks.map(k => val(A.st, A.end, k, parcF)) },
      { label: 'Com o plano', color: 'var(--blue)', values: ks.map(k => val(P.st, P.end, k, parcF)) }])}</div></div>`;

  const y0 = +from.slice(0, 4), y1 = +A.end.slice(0, 4);
  const yr = [];
  for (let y = y0; y <= y1; y++) {
    const inY = st => st.rows.filter(x => x.ym >= from && x.ym.startsWith(String(y)));
    const a = inY(A.st), b = inY(P.st);
    const ex = sum(Object.entries(P.st.rows.applied).filter(([ym]) => ym >= from && ym.startsWith(String(y))), ([, v]) => v);
    yr.push(`<tr><td>${y}</td><td class="num">${a.length ? fmt(sum(a, x => x.total) / a.length) : '—'}</td><td class="num">${b.length ? fmt(sum(b, x => x.total) / b.length) : (ex ? '—' : '<span class="pos">quitado</span>')}</td>
      <td class="num">${ex ? fmt(ex) : '—'}</td><td class="num">${fmt(a.length ? a[a.length - 1].saldo : 0)}</td><td class="num"><strong>${fmt(b.length ? b[b.length - 1].saldo : 0)}</strong></td></tr>`);
  }
  const table = `<div class="card mt"><h2>Ano a ano</h2><div class="table-wrap"><table><thead><tr><th>Ano</th><th class="num">Parcela média (hoje)</th><th class="num">Parcela média (plano)</th><th class="num">Amortização extra</th><th class="num">Saldo fim do ano (hoje)</th><th class="num">Saldo fim do ano (plano)</th></tr></thead>
    <tbody>${yr.join('')}</tbody></table></div>
    <p class="small muted">Simulação sem TR e com seguros fixos — use "🏦 Atualizar pelos dados do banco" para partir do saldo real. Confirme as regras de amortização e uso do FGTS com o banco.</p></div>`;

  return head + hero + financingStrategy(d, r) + `<div class="grid two mt">${builder}${compare}</div>` + charts + table;
}

/* ---------- inteligência: dados de mercado, estratégias de quitação e de contemplação ---------- */
const MARKET_KEY = 'minhas-financas.market';
let market = (() => { try { return JSON.parse(localStorage.getItem(MARKET_KEY)) || {}; } catch { return {}; } })();
async function fetchMarket(force = false) {
  if (!force && market.at && Date.now() - market.at < 12 * 3600e3) return;
  try {
    const get = async code => { const r = await fetch(`https://api.bcb.gov.br/dados/serie/bcdata.sgs.${code}/dados/ultimos/1?formato=json`); const j = await r.json(); return { v: parseFloat(String(j[0].valor).replace(',', '.')), d: j[0].data }; };
    const [cdi, ipca] = await Promise.allSettled([get(4389), get(13522)]);
    const next = { ...market, at: Date.now() };
    if (cdi.status === 'fulfilled' && cdi.value.v > 0) { next.cdi = cdi.value.v; next.cdiDate = cdi.value.d; }
    if (ipca.status === 'fulfilled' && ipca.value.v > -50) { next.ipca12 = ipca.value.v; next.ipcaDate = ipca.value.d; }
    if (next.cdi || next.ipca12 != null) {
      market = next;
      try { localStorage.setItem(MARKET_KEY, JSON.stringify(market)); } catch { /* */ }
      safeRender();
    }
  } catch { /* sem internet ou bloqueado: usa o valor manual */ }
}
function avgMonthly(n = 3) {
  const ms = Array.from({ length: n }, (_, i) => monthSummary(addMonths(currentMonth(), -i - 1)));
  return { surplus: sum(ms, m => m.resultado) / n, desp: sum(ms, m => m.despesas) / n };
}
const liquidBalance = () => sum(state.accounts.filter(a => a.type === 'corrente'), a => accountBalance(a));
const fmtPctNum = v => (v == null || !isFinite(v) ? '—' : String(Math.round(v * 100) / 100).replace('.', ',') + '%');

function requiredMonthly(d, from, target) {
  const ok = v => { const r = planResult(d, [{ value: v, every: 1, start: from, mode: 'prazo' }]); return r && r.P.end <= target; };
  let lo = 0, hi = financingStatus(d).saldo;
  if (!ok(hi)) return null;
  for (let k = 0; k < 32; k++) { const mid = (lo + hi) / 2; if (ok(mid)) hi = mid; else lo = mid; }
  return Math.ceil(hi / 10) * 10;
}
function fgtsNext(d, from) {
  const last = d.strategy?.fgtsLast;
  const next = last ? addMonths(last, 24) : from;
  return next < from ? from : next;
}
function strategyScenarios(d, from) {
  const g = d.strategy || {};
  const av = avgMonthly(3);
  const sugg = Math.max(0, Math.floor((av.surplus * 0.5) / 50) * 50);
  const B = g.budget !== undefined && g.budget !== '' ? +g.budget : sugg;
  if (!(B > 0)) return { B, sugg, list: [] };
  const dec = from.slice(5) === '12' ? from : `${from.slice(0, 4)}-12`;
  const fg = +g.fgtsBalance > 0 ? [{ value: +g.fgtsBalance, every: 24, start: fgtsNext(d, from), mode: 'prazo' }] : [];
  const list = [
    { key: 'mp', label: `${fmt(B)} todo mês, reduzindo o prazo`, items: [{ value: B, every: 1, start: from, mode: 'prazo' }] },
    { key: 'mc', label: `${fmt(B)} todo mês, reduzindo a parcela`, items: [{ value: B, every: 1, start: from, mode: 'parcela' }] },
    { key: 'an', label: `${fmt(B * 12)} uma vez por ano (junta e amortiza em dezembro)`, items: [{ value: B * 12, every: 12, start: dec, mode: 'prazo' }] },
  ];
  if (fg.length) list.push({ key: 'fg', label: `${fmt(B)} todo mês + FGTS a cada 2 anos`, items: [list[0].items[0], ...fg] });
  list.forEach(x => { x.res = planResult(d, x.items); });
  const best = list.reduce((a, b) => (b.res.saved > a.res.saved ? b : a));
  best.best = true;
  return { B, sugg, list };
}

function financingStrategy(d, r) {
  const g = d.strategy || {};
  const v = (k, def) => (g[k] === undefined || g[k] === '' ? def : g[k]);
  const cdi = +v('cdi', market.cdi || 0), cdiPct = +v('cdiPct', 100), ir = +v('ir', 15), tr = +v('tr', 0);
  const inp = (field, label, val, type = 'number', attrs = 'step="any"') => `<label class="field">${label}<input type="${type}" ${attrs} data-change="fs" data-id="${d.id}" data-field="${field}" value="${esc(val ?? '')}"></label>`;

  const fin = ((1 + (+d.rate || 0) / 100) * (1 + tr / 100) - 1) * 100;
  const net = cdi ? (cdi * cdiPct) / 100 * (1 - ir / 100) : null;
  const verdict = net == null ? '<span class="muted">Informe o CDI para comparar.</span>'
    : fin > net + 0.3 ? `✅ <strong>Amortizar compensa.</strong> Cada real amortizado “rende” <strong>${fmtPctNum(fin)} a.a.</strong> sem IR e sem risco, contra <strong>${fmtPctNum(net)}</strong> líquidos investindo.`
    : net > fin + 0.3 ? `📈 <strong>Investir rende mais</strong> (${fmtPctNum(net)} líquidos × ${fmtPctNum(fin)} do financiamento). Amortize com o FGTS (rende bem menos) e só com o que passar da reserva, se quiser a tranquilidade de quitar antes.`
    : `⚖️ <strong>Empate técnico</strong> (${fmtPctNum(fin)} × ${fmtPctNum(net)}). Decida pela segurança: amortizar reduz risco; investir mantém liquidez.`;

  const av = avgMonthly(3), liquid = liquidBalance();
  const months = av.desp ? liquid / av.desp : null;
  const reserve = months == null ? '<span class="muted">Sem despesas registradas nos últimos 3 meses.</span>'
    : months < 6 ? `⚠️ Você tem <strong>${fmt(liquid)}</strong> em conta ≈ <strong>${months.toFixed(1).replace('.', ',')} meses</strong> de despesas. Antes de amortizar com dinheiro do caixa, complete 6 meses (faltam <strong>${fmt(6 * av.desp - liquid)}</strong>). FGTS pode ser usado normalmente.`
    : `✅ Reserva de <strong>${months.toFixed(1).replace('.', ',')} meses</strong>. O que passa de 6 meses (<strong>${fmt(liquid - 6 * av.desp)}</strong>) pode ir para amortização. <button class="btn small soft" data-action="fs-add" data-id="${d.id}" data-value="${Math.floor((liquid - 6 * av.desp) / 100) * 100}" data-every="0">+ Amortizar esse excedente</button>`;

  const target = g.target || '';
  let targetOut = '<span class="muted">Escolha até quando quer quitar.</span>';
  if (target) {
    if (target >= r.A.end) targetOut = `Pelo contrato você já quita em ${monthLabel(r.A.end)}, antes da meta. 🎉`;
    else { const m = requiredMonthly(d, r.from, target); targetOut = m == null ? 'Meta curta demais para o saldo atual.' : `Para quitar até <strong>${monthLabel(target)}</strong>, amortize cerca de <strong class="pos">${fmt(m)} por mês</strong> (reduzindo prazo). <button class="btn small soft" data-action="fs-add" data-id="${d.id}" data-value="${m}" data-every="1">+ Adicionar ao plano</button>`; }
  }

  const fNext = fgtsNext(d, r.from);
  const fgtsOut = `Você pode usar o FGTS para amortizar a partir de <strong>${monthLabel(fNext)}</strong> (a cada 24 meses). O FGTS rende pouco (≈3% a.a. + TR), então usá-lo para amortizar quase sempre compensa.
    ${+g.fgtsBalance > 0 ? `<button class="btn small soft" data-action="fs-add" data-id="${d.id}" data-value="${+g.fgtsBalance}" data-every="24" data-start="${fNext}">+ FGTS a cada 2 anos no plano</button>` : ''}`;

  const sc = strategyScenarios(d, r.from);
  const scTable = sc.list.length ? `<div class="table-wrap"><table><thead><tr><th>Estratégia</th><th class="num">Quitação</th><th class="num">Juros economizados</th><th class="num">Próx. parcela</th><th></th></tr></thead><tbody>
    ${sc.list.map(x => `<tr class="${x.best ? 'best' : ''}"><td>${x.best ? '🏆 ' : ''}${esc(x.label)}</td><td class="num">${monthLabel(x.res.P.end)}</td><td class="num pos"><strong>${fmt(x.res.saved)}</strong></td><td class="num">${fmt(x.res.P.next)}</td>
      <td><button class="btn small ${x.best ? 'primary' : ''}" data-action="fs-use" data-id="${d.id}" data-scn="${x.key}">Usar no plano</button></td></tr>`).join('')}</tbody></table></div>` : '<p class="muted small">Informe um valor mensal para comparar as estratégias.</p>';

  return `<div class="card strat"><h2>🧠 Estratégias para quitar mais rápido</h2>
    <div class="sub">análises automáticas com seus dados · ${market.cdi ? `CDI ${fmtPctNum(market.cdi)} a.a. (Banco Central, ${esc(market.cdiDate)})` : 'CDI: informe manualmente'} <button class="btn small ghost" data-action="market-refresh">↻</button></div>
    <div class="strat-grid">
      <div class="strat-box"><h3>1. Amortizar ou investir?</h3>
        <div class="toolbar">${inp('cdi', 'CDI % a.a.', v('cdi', market.cdi || ''))}${inp('cdiPct', '% do CDI', cdiPct)}${inp('ir', 'IR %', ir)}${inp('tr', 'TR estimada % a.a.', tr)}</div>
        <p>${verdict}</p></div>
      <div class="strat-box"><h3>2. Reserva de emergência</h3><p>${reserve}</p></div>
      <div class="strat-box"><h3>3. Meta de quitação</h3>
        <div class="toolbar">${inp('target', 'Quero quitar até', target, 'month', `min="${r.from}"`)}</div><p>${targetOut}</p></div>
      <div class="strat-box"><h3>4. FGTS</h3>
        <div class="toolbar">${inp('fgtsLast', 'Último uso para amortizar', g.fgtsLast || '', 'month', '')}${inp('fgtsBalance', 'Saldo FGTS disponível (R$)', g.fgtsBalance ?? '')}</div><p>${fgtsOut}</p></div>
      <div class="strat-box full"><h3>5. Melhor forma de usar o que sobra por mês</h3>
        <div class="toolbar">${inp('budget', 'Quanto posso destinar por mês (R$)', v('budget', sc.sugg || ''))}<span class="small muted">sugestão: metade da sua sobra média dos últimos 3 meses (${fmt(av.surplus)})</span></div>
        ${scTable}</div>
    </div></div>`;
}

function consStrategyData(c) {
  const s = consorcioStatus(c), g = c.strategy || {};
  const v = (k, def) => (g[k] === undefined || g[k] === '' ? def : g[k]);
  const base = v('bidBase', 'carta') === 'saldo' ? s.remaining : s.creditNow;
  const hist = String(v('history', '')).split(/[\s;]+/).map(x => parseFloat(x.replace(',', '.'))).filter(x => x > 0 && x <= 100);
  const sorted = [...hist].sort((a, b) => a - b);
  const q = pq => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.round(pq * (sorted.length - 1)))] : null);
  const embMax = +v('embMax', 30), saved = +v('saved', 0), saving = +v('saving', 0), fgts = g.imovel ? +v('fgts', 0) : 0;
  const embR = (s.creditNow * embMax) / 100;
  const power = base ? ((saved + fgts + embR) / base) * 100 : 0;
  const rec = sorted.length ? Math.min(100, Math.ceil(q(0.75) + 1)) : null;
  return { s, g, v, base, hist, sorted, med: q(0.5), min: sorted[0], max: sorted[sorted.length - 1], rec, embMax, embR, saved, saving, fgts, power };
}

function viewConsStrategy() {
  const c = state.debts.find(x => x.id === ui.cons);
  if (!c) { ui.cons = null; return viewDebts(); }
  const k = consStrategyData(c); const { s, g, v } = k;
  const inp = (field, label, val, attrs = 'step="any" min="0"') => `<label>${label}<input type="number" ${attrs} data-change="cs" data-id="${c.id}" data-field="${field}" value="${esc(val ?? '')}"></label>`;
  const head = `<div class="page-head"><div><button class="btn small ghost" data-action="close-cons">← Voltar</button>
    <h1>Estratégia de contemplação</h1><p>${esc(c.name)} · ${esc(c.institution || '')} · carta ${fmt(s.creditNow)} · ${s.paid}/${s.total} parcelas pagas</p></div></div>`;
  if (c.contemplated) return head + '<div class="card empty-state"><h2>Já contemplado 🎉</h2></div>';

  const form = `<div class="card"><h2>Dados do grupo</h2><div class="sub">peça à administradora (app, extrato ou atendimento) e atualize de vez em quando</div>
    <div class="form-grid">
      ${inp('members', 'Cotas no grupo', v('members', ''), 'step="1" min="0"')}
      ${inp('day', 'Dia da assembleia', v('day', ''), 'step="1" min="1" max="31"')}
      ${inp('draws', 'Contemplações por sorteio/mês', v('draws', 1), 'step="1" min="0"')}
      ${inp('bids', 'Contemplações por lance/mês', v('bids', 1), 'step="1" min="0"')}
      <label class="full">Lances vencedores recentes (% — separe por espaço ou ;)
        <textarea rows="2" data-change="cs" data-id="${c.id}" data-field="history" placeholder="38; 41,5; 35; 44; 40">${esc(v('history', ''))}</textarea></label>
      <label>O lance é calculado sobre<select data-change="cs" data-id="${c.id}" data-field="bidBase"><option value="carta">a carta de crédito</option><option value="saldo" ${v('bidBase', 'carta') === 'saldo' ? 'selected' : ''}>o saldo devedor</option></select></label>
      ${inp('embMax', 'Lance embutido máximo permitido (%)', k.embMax)}
      ${inp('saved', 'Quanto já guardei para o lance (R$)', v('saved', ''))}
      ${inp('saving', 'Quanto consigo guardar por mês (R$)', v('saving', ''))}
      <label class="check full"><input type="checkbox" data-change="cs" data-id="${c.id}" data-field="imovel" ${g.imovel ? 'checked' : ''}> Consórcio de imóvel (permite usar FGTS no lance)</label>
      ${g.imovel ? inp('fgts', 'Saldo FGTS disponível (R$)', v('fgts', '')) : ''}
    </div></div>`;

  // sorteio
  const members = +v('members', 0), draws = +v('draws', 1), bids = +v('bids', 1), per = draws + bids;
  let sortOut = '<span class="muted">Informe quantas cotas tem o grupo.</span>';
  if (members && draws) {
    const rem = Math.max(1, members - s.paid * per);
    let none = 1; for (let i = 0; i < 12; i++) none *= 1 - Math.min(1, draws / Math.max(1, rem - i * per));
    sortOut = `Na próxima assembleia: <strong>${pct(Math.min(1, draws / rem))}</strong> · em 12 meses: <strong>${pct(1 - none)}</strong><br><span class="muted small">estimando ~${rem} cotas ainda não contempladas</span>`;
  }
  // histórico
  let histOut = '<span class="muted">Cole os percentuais dos lances vencedores das últimas assembleias.</span>';
  if (k.sorted.length) {
    const h = k.hist, last3 = h.slice(-3), prev = h.slice(0, -3);
    const trend = prev.length ? sum(last3) / last3.length - sum(prev) / prev.length : 0;
    histOut = `${k.sorted.length} assembleias · menor <strong>${fmtPctNum(k.min)}</strong> · mediana <strong>${fmtPctNum(k.med)}</strong> · maior <strong>${fmtPctNum(k.max)}</strong>
      ${prev.length ? `<br>Tendência: ${trend > 1 ? `📈 subindo (+${fmtPctNum(trend)} nas últimas 3)` : trend < -1 ? `📉 caindo (${fmtPctNum(trend)} nas últimas 3)` : '➡️ estável'}` : ''}
      <br>🎯 Lance recomendado: <strong>${fmtPctNum(k.rec)}</strong> = <strong>${fmt((k.base * k.rec) / 100)}</strong> <span class="muted small">(acima de 75% dos vencedores recentes)</span>`;
  }
  // poder de lance
  const wins = k.sorted.filter(x => x <= k.power).length;
  const powerOut = `Guardado ${fmt(k.saved)}${k.fgts ? ` + FGTS ${fmt(k.fgts)}` : ''} + embutido ${fmtPctNum(k.embMax)} (${fmt(k.embR)}) = <strong>${fmtPctNum(k.power)}</strong> ${k.g.bidBase === 'saldo' ? 'do saldo devedor' : 'da carta'}
    ${k.sorted.length ? `<br>Com esse lance você teria vencido em <strong>${wins} de ${k.sorted.length}</strong> assembleias recentes (${pct(wins / k.sorted.length)}).` : ''}
    <br><span class="muted small">Usando o embutido máximo, a carta liberada cai para ${fmt(s.creditNow - k.embR)}.</span>`;
  // plano
  let planOut = '<span class="muted">Informe o histórico de lances para calcular a meta.</span>';
  if (k.rec) {
    const targetOwn = (k.base * k.rec) / 100 - k.embR - k.fgts;
    if (targetOwn <= k.saved) planOut = `✅ <strong>Você já consegue ofertar o lance recomendado</strong> (${fmtPctNum(k.rec)}). Oferte na próxima assembleia.`;
    else if (k.saving > 0) {
      const i = market.cdi ? Math.pow(1 + (market.cdi * 0.85) / 100, 1 / 12) - 1 : 0;
      let acc = k.saved, m = 0; while (acc < targetOwn && m < 600) { acc = acc * (1 + i) + k.saving; m++; }
      planOut = `Faltam <strong>${fmt(targetOwn - k.saved)}</strong> do seu bolso. Guardando ${fmt(k.saving)}/mês${i ? ' num CDB/Tesouro Selic' : ''}, você chega lá em <strong>${monthsText(m)}</strong> (${monthLabel(addMonths(currentMonth(), m))}).`;
    } else planOut = `Faltam <strong>${fmt(targetOwn - k.saved)}</strong> do seu bolso. Informe quanto consegue guardar por mês para ver em quanto tempo chega lá.`;
  }
  // próxima assembleia
  const day = +v('day', 0);
  let nextAss = '';
  if (day) { let ym = currentMonth(); if (+todayISO().slice(8) > Math.min(day, daysInMonth(ym))) ym = addMonths(ym, 1); nextAss = fmtDate(dueDate(ym, day)); }
  const tips = [
    nextAss && `📅 Próxima assembleia: <strong>${nextAss}</strong>. Registre o lance alguns dias antes, no app da administradora.`,
    k.rec && `Oferte <strong>todo mês</strong> pelo menos ${fmtPctNum(Math.min(k.power, k.rec))}: se não vencer, nada é cobrado.`,
    `Combine <strong>embutido + lance livre</strong> para chegar a um percentual competitivo sem tirar tudo do bolso, lembrando que o embutido reduz a carta.`,
    `Se o grupo tiver <strong>lance fixo</strong>, participe também: é sorteado entre quem oferta o mesmo percentual.`,
    g.imovel && `No consórcio de imóvel, o <strong>FGTS</strong> pode compor o lance (confirme as regras com a administradora).`,
    `Guarde o dinheiro do lance em aplicação com liquidez diária (CDB 100% CDI / Tesouro Selic) para render até a contemplação.`,
    `Atualize aqui o histórico de lances a cada assembleia: a recomendação acompanha a disputa do grupo.`,
  ].filter(Boolean);

  const results = `<div class="stack">
    <div class="card"><h2>🎲 Chance por sorteio</h2><p>${sortOut}</p></div>
    <div class="card"><h2>📊 Lances vencedores</h2><p>${histOut}</p></div>
    <div class="card"><h2>💪 Seu poder de lance hoje</h2><p>${powerOut}</p></div>
    <div class="card"><h2>🗓️ Plano para chegar lá</h2><p>${planOut}</p></div>
  </div>`;
  return head + `<div class="grid two">${form}${results}</div>
    <div class="card mt"><h2>✅ O que fazer agora</h2><ul class="tips">${tips.map(t => `<li>${t}</li>`).join('')}</ul>
    <p class="small muted">Estimativas baseadas nos dados informados. Regras de lance variam por administradora e grupo.</p></div>`;
}

function viewDebts() {
  if (ui.cons) return viewConsStrategy();
  if (ui.sim) return viewSimulator();
  const head = `<div class="page-head"><div><h1>Financiamentos e consórcios</h1><p>Saldo devedor, parcelas, quitação, amortizações e lances</p></div>
    <div class="toolbar"><button class="btn" data-action="add-debt" data-kind="consorcio">+ Consórcio</button><button class="btn primary" data-action="add-debt" data-kind="financiamento">+ Financiamento</button></div></div>`;
  if (!state.debts.length) {
    return head + `<div class="card empty-state"><h2>Nenhum financiamento ou consórcio</h2>
      <p>Cadastre seu financiamento imobiliário (SAC ou Price) e seus consórcios para acompanhar saldo devedor, parcelas que faltam, data de quitação, quanto economiza amortizando e simular lances.</p>
      <button class="btn primary" data-action="add-debt" data-kind="financiamento">+ Financiamento</button> <button class="btn" data-action="add-debt" data-kind="consorcio">+ Consórcio</button></div>`;
  }
  const fins = state.debts.filter(d => d.kind !== 'consorcio');
  const cons = state.debts.filter(d => d.kind === 'consorcio');
  const saldoDev = sum(fins, d => financingStatus(d).saldo);
  const consLeft = sum(cons, d => consorcioStatus(d).remaining);
  const dueMonth = sum(state.debts, d => debtPaymentFor(d.id, ui.month));
  const income = monthSummary(ui.month).receitas;
  const goods = sum(fins, d => +d.assetValue || 0);
  const assets = sum(state.accounts.filter(a => a.type !== 'cartao'), a => accountBalance(a)) + portfolio().totals.value + goods;
  const cardDebt = sum(state.accounts.filter(a => a.type === 'cartao'), a => Math.max(0, -accountBalance(a)));
  const net = assets - saldoDev - cardDebt;
  const kpis = `<div class="grid kpis">
    <div class="card kpi"><div class="label">Saldo devedor (financiamentos)</div><div class="value neg">${fmt(saldoDev)}</div></div>
    <div class="card kpi"><div class="label">Parcelas em ${monthLabel(ui.month)}</div><div class="value">${fmt(dueMonth)}</div><div class="hint">${income ? pct(dueMonth / income) + ' da receita do mês' : 'sem receitas no mês'}</div></div>
    <div class="card kpi"><div class="label">Falta pagar nos consórcios</div><div class="value">${fmt(consLeft)}</div></div>
    <div class="card kpi"><div class="label">Patrimônio líquido</div><div class="value ${net >= 0 ? 'pos' : 'neg'}">${fmt(net)}</div><div class="hint">contas + investimentos${goods ? ' + bens' : ''} − dívidas</div></div>
  </div>`;
  return head + kpis + `<div class="grid debt-grid">${fins.map(financingCard).join('')}${cons.map(consorcioCard).join('')}</div>`;
}

function debtsMini() {
  if (!state.debts.length) return '';
  return `<div class="card-head mt"><h2>Financiamentos e consórcios</h2><button class="btn small" data-action="nav" data-view="debts">Ver tudo →</button></div>
    ${state.debts.map(d => { const s = debtStatus(d); return `<div class="list-row clickable" data-action="nav" data-view="debts">${debtAvatar(d)}
      <span class="grow"><strong>${esc(d.name)}</strong><br><span class="muted small">${s.paid}/${s.total} parcelas · ${d.kind === 'consorcio' ? 'falta ' + fmt(s.remaining) : 'saldo ' + fmt(s.saldo)}</span>
      <span class="bar"><span style="width:${Math.min(100, s.pct * 100)}%;background:var(--green)"></span></span></span>
      <span class="small">${s.next ? fmt(s.next.total) : '✓'}</span></div>`; }).join('')}`;
}

function debtForm(d = {}, kind) {
  const rec = d.id ? state.recurring.find(r => r.debtId === d.id) : null;
  const n = (name, label, val, attrs = '') => `<label>${label}<input type="number" step="any" name="${name}" value="${val ?? ''}" ${attrs}></label>`;
  const common = `<label class="full">Nome<input type="text" name="name" required value="${esc(d.name || '')}" placeholder="${kind === 'consorcio' ? 'Consórcio do carro' : 'Apartamento'}"></label>
    <label>${kind === 'consorcio' ? 'Administradora' : 'Banco'}<input type="text" name="institution" value="${esc(d.institution || '')}" placeholder="${kind === 'consorcio' ? 'Porto, Embracon, Itaú…' : 'Caixa'}"></label>
    <label>Vencimento da 1ª parcela<input type="date" name="firstDue" required value="${d.firstDue || todayISO()}"></label>`;
  const fields = kind === 'consorcio' ? `
    ${n('credit', 'Valor da carta de crédito (R$)', d.credit, 'required min="0"')}
    ${n('term', 'Prazo (meses)', d.term, 'required min="1" step="1"')}
    ${n('adminFee', 'Taxa de administração total (%)', d.adminFee ?? 15)}
    ${n('reserveFund', 'Fundo de reserva (%)', d.reserveFund ?? 2)}
    ${n('insurance', 'Seguro (% a.m. da carta, opcional)', d.insurance)}
    ${n('adjust', 'Reajuste anual estimado (% — INCC/IPCA)', d.adjust ?? 0)}
    ${n('paidOverride', 'Parcelas já pagas (opcional, se diferente do calendário)', d.paidOverride, 'min="0" step="1"')}
    <label class="check full"><input type="checkbox" name="contemplated" ${d.contemplated ? 'checked' : ''}> Já fui contemplado</label>
    <label>Data da contemplação<input type="date" name="contemplatedDate" value="${d.contemplatedDate || ''}"></label>
    ${n('bid', 'Lance pago (R$, abate do saldo)', d.bid)}` : `
    <label>Sistema de amortização<select name="system"><option value="SAC" ${d.system !== 'PRICE' ? 'selected' : ''}>SAC (parcelas decrescentes)</option><option value="PRICE" ${d.system === 'PRICE' ? 'selected' : ''}>Price (parcelas fixas)</option></select></label>
    ${n('principal', 'Valor financiado (R$)', d.principal, 'required min="0"')}
    ${n('rate', 'Juros efetivos (% ao ano)', d.rate, 'required min="0"')}
    ${n('term', 'Prazo total (meses)', d.term, 'required min="1" step="1"')}
    ${n('fees', 'Seguros + taxa mensal (R$ — MIP, DFI, adm.)', d.fees ?? 0)}
    ${n('assetValue', 'Valor atual do imóvel/bem (R$, opcional — entra no patrimônio)', d.assetValue)}`;
  return `<div class="form-grid">${common}${fields}
    <label class="check full"><input type="checkbox" name="autoTx" ${rec ? 'checked' : ''}> Lançar a parcela automaticamente todo mês (valor calculado pela tabela)</label>
    <label class="full">Conta de débito<select name="accountId">${accountOptions(rec?.accountId || state.accounts[0]?.id)}</select></label>
  </div>${kind === 'consorcio' ? '' : '<p class="muted small">Dica: os dados estão no contrato ou no app Habitação Caixa. A TR não é prevista — use "🏦 Atualizar pelos dados do banco" de vez em quando para ficar exato.</p>'}`;
}
function saveDebt(existing, kind, d) {
  const num = k => (d[k] === '' || d[k] === undefined ? '' : +d[k]);
  const obj = existing || { id: uid(), kind, extras: [] };
  Object.assign(obj, { name: d.name.trim(), institution: d.institution.trim(), firstDue: d.firstDue });
  if (kind === 'consorcio') Object.assign(obj, { credit: num('credit'), term: num('term'), adminFee: num('adminFee') || 0, reserveFund: num('reserveFund') || 0, insurance: num('insurance') || 0, adjust: num('adjust') || 0, paidOverride: num('paidOverride'), contemplated: d.contemplated, contemplatedDate: d.contemplatedDate, bid: num('bid') || 0 });
  else Object.assign(obj, { system: d.system, principal: num('principal'), rate: num('rate'), term: num('term'), fees: num('fees') || 0, assetValue: num('assetValue') || 0 });
  if (!existing) state.debts.push(obj);
  ensureDebtTag(kind);
  syncDebtRecurring(obj, d.autoTx, d.accountId);
  runRecurring();
}

function viewImport() {
  const p = ui.pendingImport;
  let preview = '';
  if (p) {
    const n = p.items.filter(i => i.include).length;
    preview = `<div class="card mt"><div class="card-head"><div><h2>Pré-visualização: ${esc(p.fileName)}</h2>
      <div class="sub">${p.items.length} linhas · ${p.items.filter(i => i.duplicate).length} já existentes · conta: ${esc(account(p.accountId)?.name)}${p.balance != null ? ` · saldo no arquivo: ${fmt(p.balance)}` : ''}</div></div>
      <div class="toolbar"><button class="btn" data-action="cancel-import">Cancelar</button><button class="btn primary" data-action="confirm-import">Importar ${n} transações</button></div></div>
      ${p.warnings?.length ? `<p class="muted small">⚠ ${p.warnings.map(esc).join('<br>')}</p>` : ''}
      <div class="table-wrap"><table><thead><tr><th></th><th>Data</th><th>Descrição</th><th>Tag (automática)</th><th class="num">Valor</th><th>Status</th></tr></thead><tbody>
      ${p.items.map((i, idx) => `<tr class="${i.include ? '' : 'muted'}">
        <td><input type="checkbox" data-change="imp-include" data-idx="${idx}" ${i.include ? 'checked' : ''}></td>
        <td>${fmtDate(i.date)}</td><td class="desc">${esc(i.description)}</td>
        <td><select class="tag-select" data-change="imp-tag" data-idx="${idx}">${tagOptions(i.tag)}</select></td>
        <td class="num ${i.amount >= 0 ? 'pos' : 'neg'}">${fmt(i.amount)}</td>
        <td class="small">${i.duplicate ? 'já importada' : i.maybeDup ? 'possível duplicada' : 'nova'}</td></tr>`).join('')}
      </tbody></table></div></div>`;
  }
  const pi = ui.pendingInvestImport;
  let invPreview = '';
  if (pi) {
    invPreview = `<div class="card mt"><div class="card-head"><div><h2>Operações: ${esc(pi.fileName)}</h2><div class="sub">${pi.rows.length} operações · ${pi.rows.filter(r => r.duplicate).length} já existentes</div></div>
      <div class="toolbar"><button class="btn" data-action="cancel-inv-import">Cancelar</button><button class="btn primary" data-action="confirm-inv-import">Importar ${pi.rows.filter(r => !r.duplicate).length} operações</button></div></div>
      <div class="table-wrap"><table><thead><tr><th>Data</th><th>Ativo</th><th>Classe</th><th>Tipo</th><th class="num">Qtd</th><th class="num">Preço</th><th>Status</th></tr></thead><tbody>
      ${pi.rows.map(r => `<tr class="${r.duplicate ? 'muted' : ''}"><td>${fmtDate(r.date)}</td><td>${esc(r.ticker)}</td><td>${esc(state.investments.assets.find(a => a.ticker === r.ticker)?.class || inferClass(r.ticker))}</td><td>${r.type}</td><td class="num">${fmtNum(r.qty)}</td><td class="num">${fmt(r.price)}</td><td class="small">${r.duplicate ? 'já importada' : 'nova'}</td></tr>`).join('')}
      </tbody></table></div></div>`;
  }
  return `<div class="page-head"><div><h1>Importar</h1><p>Extratos e faturas são categorizados automaticamente pelas suas regras; duplicatas são ignoradas.</p></div></div>
    <div class="grid two">
      <div class="card"><h2>Extrato bancário / fatura do cartão</h2><div class="sub">OFX (recomendado — quase todos os bancos exportam) ou CSV (Nubank, Inter, C6, Itaú…)</div>
        <div class="form-grid" style="margin-bottom:12px">
          <label>Conta de destino<select id="imp-account">${accountOptions(ui.importAccount || state.accounts[0]?.id)}</select></label>
          <label class="check" style="align-self:end"><input type="checkbox" id="imp-invert"> Inverter sinais (fatura CSV com compras positivas)</label>
        </div>
        <div class="dropzone" data-drop="bank" data-action="pick-file" data-input-id="file-bank"><strong>Arraste o arquivo aqui</strong>ou clique para escolher (.ofx, .csv, .txt)</div>
        <input type="file" id="file-bank" accept=".ofx,.qfx,.csv,.txt" hidden multiple>
      </div>
      <div class="card"><h2>Operações de investimento</h2><div class="sub">CSV da Área do Investidor B3 (Negociação) ou planilha com colunas: data, ticker, tipo, quantidade, preço, taxas</div>
        <div class="dropzone" data-drop="invest" data-action="pick-file" data-input-id="file-invest"><strong>Arraste o CSV aqui</strong>ou clique para escolher</div>
        <input type="file" id="file-invest" accept=".csv,.txt" hidden>
        <p class="muted small">Dica: na B3 acesse <em>Extratos › Negociação</em>, exporte em Excel e salve como CSV.</p>
      </div>
    </div>${preview}${invPreview}`;
}

function syncDetail() {
  const st = Sync.status(); const i = Sync.info();
  const last = i.lastSyncAt ? new Date(i.lastSyncAt).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' }) : 'nunca';
  const base = { ok: `✔ Sincronizado · última sincronização ${last}`, pending: 'Alterações pendentes — enviando em instantes…', syncing: 'Sincronizando…', error: `⚠ ${st.msg}`, off: 'Desligada' }[st.state];
  return base + (st.state === 'ok' && st.msg ? ` · ${st.msg}` : '');
}
function renderSyncStatus(st) {
  const chip = $('#sync-status');
  if (chip) {
    chip.hidden = st.state === 'off';
    chip.className = 'sync-chip ' + st.state;
    chip.textContent = { ok: '☁ Sincronizado', pending: '☁ Salvando…', syncing: '☁ Sincronizando…', error: '⚠ Sem sincronizar', off: '' }[st.state];
    chip.title = syncDetail();
  }
  const d = $('#sync-detail'); if (d) { d.textContent = syncDetail(); d.className = st.state === 'error' ? 'neg' : ''; }
  if (st.state === 'ok' && st.msg) toast(st.msg, 9000);
}
function syncCard() {
  const intro = `<h2>Sincronização entre aparelhos</h2><div class="sub">Os mesmos dados no computador e no celular. Tudo é criptografado com a sua senha <strong>antes</strong> de sair do aparelho e guardado num Gist secreto da sua conta do GitHub — nem o GitHub consegue ler.</div>`;
  if (!Sync.enabled()) {
    return `<div class="card">${intro}
      <ol class="small how-to">
        <li>Crie um token em <a href="https://github.com/settings/tokens/new?scopes=gist&description=Minhas%20Finan%C3%A7as" target="_blank" rel="noopener">github.com/settings/tokens/new</a> marcando <strong>somente a permissão "gist"</strong>. Em <em>Expiration</em>, escolha <em>No expiration</em> (ou renove quando vencer).</li>
        <li>Cole o token abaixo e crie uma <strong>senha de sincronização</strong>.</li>
        <li>Nos outros aparelhos, use <strong>o mesmo token e a mesma senha</strong>.</li>
      </ol>
      <div class="form-grid">
        <label class="full">Token do GitHub<input type="password" id="sync-token" placeholder="ghp_…" autocomplete="off"></label>
        <label class="full">Senha de sincronização<input type="password" id="sync-pass" placeholder="mínimo 8 caracteres" autocomplete="new-password"></label>
      </div>
      <div class="toolbar mt"><button class="btn primary" data-action="sync-connect">Conectar</button></div>
      <p class="muted small">⚠ Guarde a senha: sem ela os dados da nuvem não podem ser abertos (os de cada aparelho continuam lá). Token e senha ficam salvos só neste aparelho.</p></div>`;
  }
  const i = Sync.info();
  return `<div class="card">${intro}
    <p id="sync-detail" class="${Sync.status().state === 'error' ? 'neg' : ''}">${esc(syncDetail())}</p>
    <div class="toolbar"><button class="btn" data-action="sync-now">↻ Sincronizar agora</button>
      <a class="btn" href="https://gist.github.com/${esc(i.gistId)}" target="_blank" rel="noopener">Ver Gist</a>
      <button class="btn danger" data-action="sync-disconnect">Desconectar este aparelho</button></div>
    <p class="muted small">Para conectar outro aparelho: abra o app nele, vá em Configurações › Sincronização e use o mesmo token e a mesma senha.</p></div>`;
}

function viewSettings() {
  const accTypes = { corrente: 'Conta corrente / poupança', cartao: 'Cartão de crédito', corretora: 'Corretora' };
  return `<div class="page-head"><div><h1>Configurações</h1><p>Contas, tags, regras de categorização, lançamentos recorrentes e backup</p></div></div>
  <div class="grid settings-grid">
    <div class="card"><h2>Perfil e integrações</h2>
      <div class="form-grid mt">
        <label>Seu nome<input type="text" data-input="set-name" value="${esc(state.settings.name)}" placeholder="Como quer ser chamado"></label>
        <label>Token brapi.dev (cotações B3)<input type="password" data-input="set-token" value="${esc(state.settings.brapiToken)}" placeholder="opcional, gratuito"></label>
      </div>
      <p class="muted small">Sem token, a brapi só cota alguns ativos de teste. Crie um token gratuito em brapi.dev. Cripto usa a CoinGecko (sem token).</p>
    </div>
    ${syncCard()}
    <div class="card"><div class="card-head"><h2>Contas</h2><button class="btn small" data-action="add-account">+ Conta</button></div>
      ${state.accounts.map(a => `<div class="list-row">${avatar(a)}<span class="grow"><strong>${esc(a.name)}</strong><br><span class="muted small">${accTypes[a.type]} · saldo inicial ${fmt(a.initialBalance)}${a.limit ? ' · limite ' + fmt(a.limit) : ''}</span></span>
        <button class="icon-btn" data-action="edit-account" data-id="${a.id}">✎</button><button class="icon-btn" data-action="del-account" data-id="${a.id}">✕</button></div>`).join('')}
    </div>
    <div class="card"><div class="card-head"><div><h2>Regras automáticas</h2><div class="sub">se a descrição contém… então aplicar a tag (primeira regra que casar vence)</div></div><button class="btn small" data-action="add-rule">+ Regra</button></div>
      ${state.rules.map((r, i) => `<div class="list-row"><span class="dot" style="background:${tagInfo(r.tag).color}"></span><span class="grow small" title="${esc(r.pattern)}">${esc(r.pattern)}</span>
        <span class="pill">${esc(r.tag)}${r.sign === 'in' ? ' · entradas' : r.sign === 'out' ? ' · saídas' : ''}</span>
        <button class="icon-btn" data-action="rule-up" data-idx="${i}" title="Subir prioridade">↑</button>
        <button class="icon-btn" data-action="edit-rule" data-id="${r.id}">✎</button><button class="icon-btn" data-action="del-rule" data-id="${r.id}">✕</button></div>`).join('')}
    </div>
    <div class="card"><div class="card-head"><div><h2>Recorrentes</h2><div class="sub">lançados automaticamente todo mês (salário, aluguel, assinaturas…)</div></div><button class="btn small" data-action="add-rec">+ Recorrente</button></div>
      ${state.recurring.length ? state.recurring.map(r => `<div class="list-row"><span class="grow"><strong>${esc(r.description)}</strong><br><span class="muted small">todo dia ${r.day} · ${esc(r.tag || 'Sem Tags')} · desde ${monthLabel(r.startMonth)}${r.endMonth ? ' até ' + monthLabel(r.endMonth) : ''}</span></span>
        <span class="${r.amount >= 0 ? 'pos' : 'neg'}">${fmt(r.amount)}</span><button class="icon-btn" data-action="edit-rec" data-id="${r.id}">✎</button><button class="icon-btn" data-action="del-rec" data-id="${r.id}">✕</button></div>`).join('') : '<p class="muted">Nenhum lançamento recorrente.</p>'}
    </div>
    <div class="card"><div class="card-head"><h2>Tags</h2><button class="btn small" data-action="add-tag">+ Tag</button></div>
      ${state.tags.map(t => `<div class="list-row">${iconBadge(t.icon, t.color)}<span class="grow">${esc(t.name)}${t.exclude ? ' <span class="muted small">(fora de receitas/despesas)</span>' : ''}${t.group ? ` <span class="muted small">· ${GROUPS[t.group]}</span>` : ''}</span>
        <button class="icon-btn" data-action="edit-tag" data-name="${esc(t.name)}">✎</button><button class="icon-btn" data-action="del-tag" data-name="${esc(t.name)}">✕</button></div>`).join('')}
    </div>
    <div class="card"><h2>Backup</h2><div class="sub">Os dados ficam só neste navegador. Exporte com frequência.</div>
      <div class="toolbar mt"><button class="btn" data-action="export">⬇ Exportar JSON</button><button class="btn" data-action="pick-file" data-input-id="file-backup">⬆ Restaurar JSON</button>
        <input type="file" id="file-backup" accept=".json" hidden></div>
      <div class="toolbar"><button class="btn" data-action="load-demo">Carregar dados de exemplo</button><button class="btn danger" data-action="reset">Apagar tudo</button></div>
      ${(() => { const c = Sync.conflictBackup(); return c ? `<p class="small muted">Backup de conflito de sincronização (versão ${esc(c.origin)}, ${new Date(c.at).toLocaleString('pt-BR')}): <button class="btn small" data-action="restore-conflict">Restaurar</button> <button class="btn small ghost" data-action="clear-conflict">Descartar</button></p>` : ''; })()}
    </div>
  </div>`;
}

const VIEWS = { overview: viewOverview, transactions: viewTransactions, investments: viewInvestments, debts: viewDebts, import: viewImport, settings: viewSettings };
function render() {
  document.querySelectorAll('#tabs button, #bottom-nav button[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === ui.view));
  $('#hide-btn').textContent = prefs.hide ? '🙈' : '👁';
  $('#app').innerHTML = VIEWS[ui.view]();
}

/* =============================================================== modal/toast */
let modalSubmit = null;
function openModal(title, body, onSubmit, { submitLabel = 'Salvar', onOpen, noSubmit = false } = {}) {
  const dlg = $('#modal');
  dlg.querySelector('.modal-title').textContent = title;
  dlg.querySelector('.modal-body').innerHTML = body;
  dlg.querySelector('button[type=submit]').textContent = submitLabel;
  dlg.querySelector('button[type=submit]').hidden = noSubmit;
  modalSubmit = onSubmit;
  if (!dlg.open) dlg.showModal();
  onOpen?.(dlg);
}
$('#modal-form').addEventListener('submit', e => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target));
  e.target.querySelectorAll('input[type=checkbox]').forEach(c => { data[c.name] = c.checked; });
  if (modalSubmit && modalSubmit(data) === false) return;
  $('#modal').close(); save(); render();
});

let toastTimer;
function toast(msg, ms = 3500) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

/* ================================================================== forms */
function txForm(t = {}) {
  return `<div class="form-grid">
    <label>Data<input type="date" name="date" required value="${t.date || todayISO()}"></label>
    <label>Valor (negativo = despesa)<input type="number" step="0.01" name="amount" required value="${t.amount ?? ''}" placeholder="-50.00"></label>
    <label class="full">Descrição<input type="text" name="description" required value="${esc(t.description || '')}"></label>
    <label>Tag<select name="tag">${tagOptions(t.tag || '')}</select></label>
    <label>Conta<select name="accountId">${accountOptions(t.accountId || state.accounts[0]?.id)}</select></label>
    <label class="check full"><input type="checkbox" name="ignore" ${t.ignore ? 'checked' : ''}> Ignorar nos totais de receitas/despesas</label>
  </div>`;
}

function ruleForm(r = {}) {
  return `<div class="form-grid">
    <label class="full">Se a descrição contém (separe alternativas com |)<input type="text" name="pattern" required value="${esc(r.pattern || '')}" placeholder="ifood|rappi"></label>
    <label>Aplicar a tag<select name="tag" required>${tagOptions(r.tag || '', { none: false })}</select></label>
    <label>Somente para<select name="sign">
      <option value="any" ${r.sign === 'any' ? 'selected' : ''}>Entradas e saídas</option>
      <option value="out" ${r.sign === 'out' ? 'selected' : ''}>Saídas</option>
      <option value="in" ${r.sign === 'in' ? 'selected' : ''}>Entradas</option></select></label>
    <label class="check full"><input type="checkbox" name="applyAll" checked> Aplicar agora às transações sem tag que casarem</label>
  </div>`;
}
function saveRule(existing, d) {
  const r = existing || { id: uid() };
  Object.assign(r, { pattern: d.pattern.trim(), tag: d.tag, sign: d.sign });
  if (!existing) state.rules.unshift(r);
  if (d.applyAll) {
    let n = 0;
    for (const t of state.transactions) if (!t.tag) { const s = suggestTag(t.description, t.amount); if (s) { t.tag = s; n++; } }
    if (n) toast(`${n} transações categorizadas.`);
  }
}

function opForm(ticker = '') {
  return `<div class="form-grid">
    <label>Ativo (ticker)<input type="text" name="ticker" required list="tickers" value="${esc(ticker)}" placeholder="PETR4, MXRF11, BTC, CDB Banco X…"><datalist id="tickers">${state.investments.assets.map(a => `<option value="${esc(a.ticker)}">`).join('')}</datalist></label>
    <label>Classe (se novo)<select name="cls"><option value="auto">Detectar automaticamente</option>${Object.keys(ASSET_CLASSES).map(c => `<option>${c}</option>`).join('')}</select></label>
    <label>Tipo<select name="type"><option value="compra">Compra / aporte</option><option value="venda">Venda / resgate</option><option value="provento">Provento (dividendo, JCP, rendimento)</option></select></label>
    <label>Data<input type="date" name="date" required value="${todayISO()}"></label>
    <label data-f="qty">Quantidade<input type="number" step="any" name="qty" value="1"></label>
    <label data-f="price"><span>Preço unitário (R$)</span><input type="number" step="any" name="price" required></label>
    <label data-f="fees">Taxas / corretagem (R$)<input type="number" step="any" name="fees" value="0"></label>
    <label data-f="idx">Rende<select name="indexer">${Object.entries(INDEXERS).map(([k, l]) => `<option value="${k}" ${k === 'cdi' ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
    <label data-f="idx">Taxa (110 = 110% CDI; 6 = IPCA+6%)<input type="number" step="any" name="idxRate" placeholder="100"></label>
    <label class="check full"><input type="checkbox" name="createTx"> Lançar também como transação na conta</label>
    <label class="full" data-f="acc">Conta<select name="accountId">${accountOptions(state.accounts[0]?.id)}</select></label>
  </div>`;
}
function wireOpForm(dlg) {
  const f = dlg.querySelector('form');
  const sync = () => {
    const ticker = f.ticker.value.trim().toUpperCase();
    const existing = state.investments.assets.find(a => a.ticker === ticker);
    const cls = existing ? existing.class : (f.cls.value === 'auto' ? inferClass(ticker) : f.cls.value);
    const manual = ASSET_CLASSES[cls]?.manual;
    const prov = f.type.value === 'provento';
    f.querySelector('[data-f=qty]').style.display = manual || prov ? 'none' : '';
    f.querySelector('[data-f=price] span').textContent = manual || prov ? 'Valor total (R$)' : 'Preço unitário (R$)';
    f.querySelector('[data-f=acc]').style.display = f.createTx.checked ? '' : 'none';
    f.querySelectorAll('[data-f=idx]').forEach(x => { x.style.display = manual && !existing ? '' : 'none'; });
    f.cls.disabled = !!existing;
  };
  ['input', 'change'].forEach(ev => f.addEventListener(ev, sync));
  sync();
}
function saveOp(d) {
  const isNew = !state.investments.assets.some(x => x.ticker === d.ticker.trim().toUpperCase());
  const a = findOrCreateAsset(d.ticker, d.cls);
  const manual = ASSET_CLASSES[a.class]?.manual;
  if (isNew && manual && d.indexer && +d.idxRate > 0) { a.indexer = d.indexer; a.idxRate = +d.idxRate; }
  let qty = +d.qty || 0, price = +d.price || 0;
  if (manual) { qty = price; price = 1; }        // renda fixa: quantidade = R$ aplicado, preço unitário 1
  else if (d.type === 'provento') { qty = 1; }
  if (!qty || !price) { toast('Informe quantidade e preço.'); return false; }
  if (d.type === 'venda' && manual && a.manualValue != null && a.manualValue !== '') a.manualValue = Math.max(0, +a.manualValue - qty);
  if (d.type === 'compra' && manual && a.manualValue != null && a.manualValue !== '') a.manualValue = +a.manualValue + qty;
  state.investments.operations.push({ id: uid(), assetId: a.id, date: d.date, type: d.type, qty, price, fees: +d.fees || 0 });
  if (d.createTx) {
    const total = qty * price;
    const amount = d.type === 'compra' ? -(total + (+d.fees || 0)) : total - (+d.fees || 0);
    const label = { compra: 'Aporte', venda: 'Resgate', provento: 'Provento' }[d.type];
    state.transactions.push({ id: uid(), date: d.date, description: `${label} ${a.ticker}`, amount: round2(amount), tag: 'Investimentos', accountId: d.accountId, source: 'investimento' });
  }
  recordSnapshot();
}

/* ================================================================ import */
function readFile(file) { return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsText(file, guessEncoding(file)); }); }
function guessEncoding() { return 'utf-8'; }
async function readText(file) {
  // Muitos bancos exportam em latin1 (windows-1252); se aparecer caractere inválido, relê nesse encoding.
  let text = await readFile(file);
  if (text.includes('�')) text = await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsText(file, 'windows-1252'); });
  return text;
}

async function handleBankFiles(files) {
  const accountId = $('#imp-account')?.value || state.accounts[0].id;
  ui.importAccount = accountId;
  const acc = account(accountId);
  const invert = $('#imp-invert')?.checked;
  let rows = [], warnings = [], balance = null, names = [];
  for (const file of files) {
    const text = await readText(file);
    names.push(file.name);
    if (/<OFX>|OFXHEADER|<STMTTRN>/i.test(text)) {
      const r = Importers.parseOFX(text); rows.push(...r.rows); balance = r.balance;
    } else {
      const r = Importers.parseBankCSV(text); warnings.push(...r.warnings);
      // Fatura CSV de cartão (ex.: Nubank) traz compras positivas: inverte automaticamente se a conta é cartão.
      const flip = invert || (acc.type === 'cartao' && r.rows.length && r.rows.filter(x => x.amount > 0).length > r.rows.length / 2);
      if (flip) r.rows.forEach(x => { x.amount = -x.amount; });
      if (flip && !invert) warnings.push('Sinais invertidos automaticamente (fatura de cartão com compras positivas).');
      rows.push(...r.rows);
    }
  }
  if (!rows.length) { toast('Nenhuma transação encontrada. ' + warnings.join(' '), 7000); return; }
  const existing = new Set(state.transactions.map(t => t.externalId).filter(Boolean));
  const others = state.transactions.filter(t => t.accountId === accountId && !String(t.externalId || '').startsWith(accountId + ':'));
  const seen = {};
  const items = rows.map(r => {
    const amount = round2(r.amount);
    let ext;
    if (r.id) ext = `${accountId}:${r.id}`;
    else { const base = `${accountId}:${r.date}:${amount.toFixed(2)}:${norm(r.description)}`; seen[base] = (seen[base] || 0) + 1; ext = `${base}:${seen[base]}`; }
    const duplicate = existing.has(ext);
    const maybeDup = !duplicate && others.some(t => t.date === r.date && Math.abs(t.amount - amount) < 0.005);
    const catTag = r.category && state.tags.find(t => norm(t.name) === norm(r.category))?.name;
    return { date: r.date, description: r.description, amount, externalId: ext, duplicate, maybeDup, include: !duplicate && !maybeDup, tag: catTag || suggestTag(r.description, amount) };
  }).sort((a, b) => b.date.localeCompare(a.date));
  ui.pendingImport = { items, accountId, fileName: names.join(', '), warnings, balance };
  render();
}

async function handleInvestFile(file) {
  const text = await readText(file);
  const { rows, warnings } = Importers.parseInvestCSV(text);
  if (!rows.length) { toast('Nenhuma operação encontrada. ' + warnings.join(' '), 7000); return; }
  const key = (date, ticker, type, qty, price) => `${date}|${ticker}|${type}|${(+qty).toFixed(6)}|${(+price).toFixed(4)}`;
  const existing = new Set(state.investments.operations.map(o => key(o.date, state.investments.assets.find(a => a.id === o.assetId)?.ticker, o.type, o.qty, o.price)));
  rows.forEach(r => { r.duplicate = existing.has(key(r.date, r.ticker, r.type, r.qty, r.price)); });
  ui.pendingInvestImport = { rows, fileName: file.name };
  render();
}

/* ================================================================ demo data */
function loadDemo() {
  if ((state.transactions.length || Sync.enabled()) && !confirm('Isso vai substituir seus dados atuais pelos dados de exemplo' + (Sync.enabled() ? ' — também na nuvem e nos outros aparelhos sincronizados' : '') + '. Continuar?')) return;
  let seed = 42; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const s = defaultState();
  s.settings.name = state.settings.name;
  s.settings.brapiToken = state.settings.brapiToken;
  s.accounts = [
    { id: 'conta', name: 'Nubank', type: 'corrente', initialBalance: 8500, color: '#7c3aed' },
    { id: 'poupanca', name: 'Banco do Brasil', type: 'corrente', initialBalance: 6000, color: '#0ea5e9' },
    { id: 'cartao', name: 'Itaú Visa', type: 'cartao', initialBalance: -2600, limit: 9000, color: '#e11d48' },
  ];
  s.budgets = { 'Alimentação': 1600, 'Transporte': 600, 'Lazer': 350, 'Compras': 300, 'Assinaturas': 100 };
  const tx = (date, description, amount, accountId = 'conta') => s.transactions.push({ id: uid(), date, description, amount: round2(amount), tag: '', accountId, source: 'exemplo' });
  const end = todayISO();
  for (let i = -6; i <= 0; i++) {
    const m = addMonths(currentMonth(), i);
    const d = n => `${m}-${pad(Math.min(n, daysInMonth(m)))}`;
    const add = (n, ...a) => { if (d(n) <= end) tx(d(n), ...a); };
    add(5, 'SALARIO EMPRESA XYZ LTDA', 12500);
    add(6, 'TED XP INVESTIMENTOS CORRETORA', -2500);
    add(10, 'ALUGUEL APTO', -3200);
    add(12, 'CONDOMINIO EDIFICIO', -780);
    add(15, 'ENEL DISTRIBUICAO', -180 - rnd() * 90);
    add(16, 'VIVO FIBRA INTERNET', -119.9);
    add(8, 'PAGAMENTO DE FATURA', -2600 - rnd() * 900);
    add(8, 'PAGAMENTO RECEBIDO FATURA', 2600, 'cartao');
    add(20, 'RENDIMENTO POUPANCA', 35 + rnd() * 20);
    add(3, 'NETFLIX.COM', -55.9, 'cartao');
    add(4, 'SPOTIFY', -21.9, 'cartao');
    add(22, 'FARMACIA DROGASIL', -40 - rnd() * 120, 'cartao');
    add(18, 'CURSO ALURA', -99, 'cartao');
    for (let k = 0; k < 4; k++) add(2 + k * 7, 'SUPERMERCADO PAO DE ACUCAR', -180 - rnd() * 260, 'cartao');
    for (let k = 0; k < 8; k++) add(1 + Math.floor(rnd() * 28), 'IFOOD *RESTAURANTE', -35 - rnd() * 60, 'cartao');
    for (let k = 0; k < 10; k++) add(1 + Math.floor(rnd() * 28), 'UBER *TRIP', -14 - rnd() * 35, 'cartao');
    for (let k = 0; k < 2; k++) add(1 + Math.floor(rnd() * 28), 'POSTO SHELL', -150 - rnd() * 100, 'cartao');
    add(1 + Math.floor(rnd() * 27), 'CINEMARK INGRESSO', -60 - rnd() * 40, 'cartao');
    add(1 + Math.floor(rnd() * 27), 'BAR DO ZE', -80 - rnd() * 150, 'cartao');
    add(1 + Math.floor(rnd() * 27), 'AMAZON MARKETPLACE', -70 - rnd() * 400, 'cartao');
    add(1 + Math.floor(rnd() * 27), 'PIX ENVIADO JOAO SILVA', -50 - rnd() * 200);
  }
  // Pagamento de fatura = gastos do cartão no mês anterior
  for (const t of s.transactions) if (/PAGAMENTO (DE|RECEBIDO) FATURA/.test(t.description)) {
    const prev = addMonths(t.date.slice(0, 7), -1);
    const spent = -sum(s.transactions.filter(x => x.accountId === 'cartao' && x.amount < 0 && x.date.startsWith(prev)), x => x.amount);
    t.amount = round2(t.description.includes('RECEBIDO') ? (spent || 2600) : -(spent || 2600));
  }
  s.transactions.forEach(t => { t.tag = suggestTagWith(s.rules, t.description, t.amount); });
  s.recurring.push({ id: uid(), description: 'SALARIO EMPRESA XYZ LTDA', amount: 12500, tag: 'Salário', accountId: 'conta', day: 5, startMonth: addMonths(currentMonth(), 1), generated: [] });

  const inv = s.investments;
  const asset = (ticker, cls, currentPrice, extra = {}) => { const a = { id: uid(), ticker, class: cls, currentPrice, manualValue: null, quoteId: '', ...extra }; inv.assets.push(a); return a; };
  const op = (a, months, type, qty, price) => inv.operations.push({ id: uid(), assetId: a.id, date: `${addMonths(currentMonth(), months)}-06`, type, qty, price, fees: 0 });
  const petr = asset('PETR4', 'Ações', 38.42, { high52: 42.1, low52: 31.5 }), itub = asset('ITUB4', 'Ações', 36.1, { high52: 37.2, low52: 29.8 }), vale = asset('VALE3', 'Ações', 61.3, { high52: 79.9, low52: 55.2 });
  const mxrf = asset('MXRF11', 'FIIs', 9.72, { high52: 10.4, low52: 9.3 }), hglg = asset('HGLG11', 'FIIs', 158.4, { high52: 171, low52: 150 }), ivvb = asset('IVVB11', 'ETFs', 352.1, { high52: 360, low52: 290 });
  const selic = asset('TESOURO SELIC 2029', 'Tesouro Direto', null, { indexer: 'cdi', idxRate: 100, manualValue: 15840 });
  const cdb = asset('CDB BANCO INTER 110% CDI', 'Renda Fixa', null, { indexer: 'cdi', idxRate: 110, manualValue: 10920 });
  const btc = asset('BTC', 'Cripto', 512000, { quoteId: 'bitcoin' });
  inv.targets = { 'Renda Fixa': 20, 'Tesouro Direto': 20, 'Ações': 25, 'FIIs': 15, 'ETFs': 10, 'Cripto': 10 };
  op(petr, -11, 'compra', 100, 34.1); op(petr, -5, 'compra', 50, 36.8); op(itub, -9, 'compra', 200, 31.2); op(vale, -7, 'compra', 60, 66.5);
  op(mxrf, -10, 'compra', 600, 10.2); op(mxrf, -3, 'compra', 300, 9.6); op(hglg, -8, 'compra', 25, 162); op(ivvb, -6, 'compra', 15, 318);
  op(selic, -12, 'compra', 10000, 1); op(selic, -4, 'compra', 5000, 1); op(cdb, -9, 'compra', 10000, 1);
  op(btc, -10, 'compra', 0.02, 380000); op(btc, -2, 'compra', 0.01, 470000);
  for (let i = -9; i <= 0; i++) { op(mxrf, i, 'provento', 1, 6.5 + (i > -3 ? 2.7 : 0) * 1); op(hglg, i, 'provento', 1, 27.5); }
  op(petr, -4, 'provento', 1, 210); op(itub, -2, 'provento', 1, 95); op(petr, -5, 'venda', 30, 39.9);
  state = s;
  ensureDebtTag('financiamento'); ensureDebtTag('consorcio');
  state.debts = [
    { id: uid(), kind: 'financiamento', name: 'Apartamento', institution: 'Caixa', system: 'SAC', principal: 320000, rate: 9.5, term: 360, fees: 110, assetValue: 460000, firstDue: `${addMonths(currentMonth(), -47)}-15`, extras: [{ date: `${addMonths(currentMonth(), -10)}-20`, value: 15000, mode: 'prazo' }] },
    { id: uid(), kind: 'consorcio', name: 'Consórcio do carro', institution: 'Porto Consórcio', credit: 90000, term: 72, adminFee: 14, reserveFund: 2, insurance: 0, adjust: 4, paidOverride: '', contemplated: false, firstDue: `${addMonths(currentMonth(), -20)}-10`, bid: 0 },
    { id: uid(), kind: 'consorcio', name: 'Consórcio imóvel', institution: 'Itaú Consórcio', credit: 300000, term: 200, adminFee: 19, reserveFund: 2, insurance: 0, adjust: 5, paidOverride: '', contemplated: true, contemplatedDate: `${addMonths(currentMonth(), -3)}-12`, firstDue: `${addMonths(currentMonth(), -14)}-20`, bid: 45000 },
  ];
  const total = portfolio().totals.value;
  for (let i = -11; i <= 0; i++) inv.history[addMonths(currentMonth(), i)] = round2(total * (0.62 + 0.38 * ((i + 11) / 11)) * (0.97 + rnd() * 0.06));
  inv.history[currentMonth()] = round2(total);
  save(); render();
  toast('Dados de exemplo carregados.');
  autoQuotes(true);
}
function suggestTagWith(rules, desc, amount) { const old = state.rules; state.rules = rules; const r = suggestTag(desc, amount); state.rules = old; return r; }

/* ================================================================ actions */
const actions = {
  nav: el => { if ($('#modal').open) $('#modal').close(); ui.sim = null; ui.cons = null; ui.analysis = false; ui.view = el.dataset.view; render(); window.scrollTo(0, 0); },
  'toggle-hide': () => { prefs.hide = !prefs.hide; savePrefs(); render(); },
  'hide-onboarding': () => { prefs.hideOnboarding = true; savePrefs(); render(); },
  'onboarding-go': el => { if (el.dataset.go === 'budgets') return actions['edit-budgets'](); ui.view = el.dataset.go; render(); window.scrollTo(0, 0); },
  'goto-account': el => { ui.filters = { month: '__all', q: '', tag: '__all', account: el.dataset.id, day: '' }; ui.view = 'transactions'; render(); window.scrollTo(0, 0); },
  'quick-menu': () => openModal('O que você quer lançar?', quickActions(true), null, { noSubmit: true }),
  'quick-tx': el => {
    const out = el.dataset.kind === 'out';
    openModal(out ? 'Nova despesa' : 'Nova receita', quickTxForm(out), d => {
      const v = Math.abs(+d.value); if (!v) { toast('Informe o valor.'); return false; }
      const amount = round2(out ? -v : v);
      state.transactions.push({ id: uid(), date: d.date, description: d.description.trim(), amount, tag: d.tag || suggestTag(d.description, amount), accountId: d.accountId, source: 'manual' });
      toast(out ? 'Despesa lançada.' : 'Receita lançada.');
    });
  },
  transfer: () => openModal('Transferência entre contas', transferForm(), d => {
    if (d.from === d.to) { toast('Escolha contas diferentes.'); return false; }
    const v = round2(Math.abs(+d.value)); if (!v) { toast('Informe o valor.'); return false; }
    const from = account(d.from), to = account(d.to);
    state.transactions.push(
      { id: uid(), date: d.date, description: `Transferência para ${to.name}`, amount: -v, tag: 'Transferência', accountId: from.id, source: 'manual' },
      { id: uid(), date: d.date, description: `Transferência de ${from.name}`, amount: v, tag: 'Transferência', accountId: to.id, source: 'manual' });
  }),
  'adjust-balances': () => openModal('Ajustar saldos',
    `<p class="muted small">Informe o saldo que aparece hoje no app de cada banco (para cartões, o valor da fatura atual). O saldo inicial da conta é corrigido para bater, sem criar transações.</p>
    <div class="form-grid">${state.accounts.map(a => `<label>${esc(a.name)}${a.type === 'cartao' ? ' — fatura atual' : ''}<input type="number" step="0.01" name="acc_${a.id}" value="${round2(a.type === 'cartao' ? Math.max(0, -accountBalance(a)) : accountBalance(a))}"></label>`).join('')}</div>`,
    d => {
      for (const a of state.accounts) {
        const v = d['acc_' + a.id]; if (v === '' || v === undefined) continue;
        const target = a.type === 'cartao' ? -Math.abs(+v) : +v;
        a.initialBalance = round2((+a.initialBalance || 0) + target - accountBalance(a));
      }
      toast('Saldos ajustados.');
    }),
  'edit-budgets': () => {
    const tags = state.tags.filter(t => !t.exclude && (t.group === 'essencial' || t.group === 'livre'));
    openModal('Metas mensais de gastos', `<p class="muted small">Deixe em branco as tags sem meta. A dica em cada campo é sua média dos últimos 3 meses.</p>
      <div class="form-grid">${tags.map((t, i) => { const avg = avgSpent(t.name, currentMonth()); return `<label><span><span class="dot" style="background:${t.color}"></span> ${esc(t.name)}</span>
        <input type="number" step="1" min="0" name="b_${i}" value="${state.budgets?.[t.name] ?? ''}" placeholder="${avg ? 'média ' + Math.round(avg) : ''}"></label>`; }).join('')}
        <label class="check full"><input type="checkbox" name="fillAvg"> Preencher as vazias com a média dos últimos 3 meses</label></div>`, d => {
      const b = {};
      tags.forEach((t, i) => {
        let v = d['b_' + i];
        if ((v === '' || v === undefined) && d.fillAvg) { const avg = avgSpent(t.name, currentMonth()); if (avg > 0) v = Math.ceil(avg / 10) * 10; }
        if (v !== '' && v !== undefined && +v > 0) b[t.name] = +v;
      });
      state.budgets = b;
    });
  },
  'month-prev': () => { ui.month = addMonths(ui.month, -1); render(); },
  'month-next': () => { ui.month = addMonths(ui.month, 1); render(); },
  donut: el => { ui.donutMode = el.dataset.mode; render(); },
  'goto-day': el => { ui.filters = { ...ui.filters, day: el.dataset.day, tag: '__all', q: '' }; ui.view = 'transactions'; render(); },
  'goto-tag': el => { if (!el.dataset.tag && el.dataset.tag !== '') return; ui.filters = { ...ui.filters, month: ui.month, day: '', tag: el.dataset.tag, q: '' }; ui.view = 'transactions'; render(); },
  'clear-day': () => { ui.filters.month = ui.filters.day.slice(0, 7); ui.filters.day = ''; render(); },
  'load-demo': loadDemo,

  'add-tx': () => openModal('Nova transação', txForm(), d => {
    state.transactions.push({ id: uid(), date: d.date, description: d.description.trim(), amount: round2(+d.amount), tag: d.tag || suggestTag(d.description, +d.amount), accountId: d.accountId, ignore: d.ignore, source: 'manual' });
  }),
  'edit-tx': el => { const t = state.transactions.find(x => x.id === el.dataset.id); openModal('Editar transação', txForm(t), d => { Object.assign(t, { date: d.date, description: d.description.trim(), amount: round2(+d.amount), tag: d.tag, accountId: d.accountId, ignore: d.ignore }); }); },
  'del-tx': el => { state.transactions = state.transactions.filter(t => t.id !== el.dataset.id); save(); render(); },
  'apply-rules': () => {
    let n = 0;
    for (const t of state.transactions) if (!t.tag) { const s = suggestTag(t.description, t.amount); if (s) { t.tag = s; n++; } }
    save(); render(); toast(n ? `${n} transações categorizadas.` : 'Nenhuma regra casou. Crie regras com ⚡ na transação.');
  },
  'rule-from-tx': el => {
    const t = state.transactions.find(x => x.id === el.dataset.id);
    const pattern = norm(t.description).replace(/[*#]/g, ' ').replace(/\d{2,}/g, ' ').replace(/\s+/g, ' ').trim().split(' ').slice(0, 3).join(' ');
    openModal('Criar regra automática', ruleForm({ pattern, tag: t.tag || state.tags[0].name, sign: t.amount < 0 ? 'out' : 'in' }), d => saveRule(null, d));
  },

  'add-rule': () => openModal('Nova regra', ruleForm({ sign: 'out' }), d => saveRule(null, d)),
  'edit-rule': el => { const r = state.rules.find(x => x.id === el.dataset.id); openModal('Editar regra', ruleForm(r), d => saveRule(r, d)); },
  'del-rule': el => { state.rules = state.rules.filter(r => r.id !== el.dataset.id); save(); render(); },
  'rule-up': el => { const i = +el.dataset.idx; if (i > 0) { [state.rules[i - 1], state.rules[i]] = [state.rules[i], state.rules[i - 1]]; save(); render(); } },

  'add-account': () => openModal('Nova conta', accountForm(), d => { state.accounts.push({ id: uid(), name: d.name, type: d.type, initialBalance: +d.initialBalance || 0, limit: +d.limit || 0, color: d.color, bank: d.bank, logo: d.logo || '' }); }, { onOpen: wireAccountForm }),
  'edit-account': el => { const a = account(el.dataset.id); openModal('Editar conta', accountForm(a), d => { Object.assign(a, { name: d.name, type: d.type, initialBalance: +d.initialBalance || 0, limit: +d.limit || 0, color: d.color, bank: d.bank, logo: d.logo || '' }); }, { onOpen: wireAccountForm }); },
  'del-account': el => {
    if (state.accounts.length === 1) return toast('Mantenha pelo menos uma conta.');
    const n = state.transactions.filter(t => t.accountId === el.dataset.id).length;
    if (!confirm(`Excluir a conta${n ? ` e suas ${n} transações` : ''}?`)) return;
    state.accounts = state.accounts.filter(a => a.id !== el.dataset.id);
    state.transactions = state.transactions.filter(t => t.accountId !== el.dataset.id);
    save(); render();
  },

  'add-tag': () => openModal('Nova tag', tagForm(), d => { if (state.tags.some(t => t.name === d.name.trim())) { toast('Já existe.'); return false; } state.tags.push({ name: d.name.trim(), color: d.color, icon: d.icon.trim() || '🏷️', group: d.group, exclude: d.exclude }); }),
  'edit-tag': el => {
    const t = state.tags.find(x => x.name === el.dataset.name); const old = t.name;
    openModal('Editar tag', tagForm(t), d => {
      const name = d.name.trim(); Object.assign(t, { name, color: d.color, icon: d.icon.trim() || '🏷️', group: d.group, exclude: d.exclude });
      if (name !== old && state.budgets?.[old] !== undefined) { state.budgets[name] = state.budgets[old]; delete state.budgets[old]; }
      if (name !== old) { state.transactions.forEach(x => { if (x.tag === old) x.tag = name; }); state.rules.forEach(x => { if (x.tag === old) x.tag = name; }); state.recurring.forEach(x => { if (x.tag === old) x.tag = name; }); }
    });
  },
  'del-tag': el => {
    const name = el.dataset.name;
    if (!confirm(`Excluir a tag "${name}"? As transações ficam sem tag.`)) return;
    state.tags = state.tags.filter(t => t.name !== name);
    state.transactions.forEach(t => { if (t.tag === name) t.tag = ''; });
    state.rules = state.rules.filter(r => r.tag !== name);
    save(); render();
  },

  'add-rec': () => openModal('Novo lançamento recorrente', recForm(), d => { state.recurring.push({ id: uid(), ...recFromForm(d), generated: [] }); runRecurring(); }),
  'edit-rec': el => { const r = state.recurring.find(x => x.id === el.dataset.id); openModal('Editar recorrente', recForm(r), d => { Object.assign(r, recFromForm(d)); runRecurring(); }); },
  'del-rec': el => { state.recurring = state.recurring.filter(r => r.id !== el.dataset.id); save(); render(); },

  'add-op': el => openModal('Nova operação', opForm(el.dataset.ticker || ''), saveOp, { onOpen: wireOpForm }),
  'del-op': el => { state.investments.operations = state.investments.operations.filter(o => o.id !== el.dataset.id); recordSnapshot(); save(); render(); },
  'edit-asset': el => {
    const a = state.investments.assets.find(x => x.id === el.dataset.id);
    const manual = ASSET_CLASSES[a.class]?.manual;
    openModal(`Editar ${a.ticker}`, `<div class="form-grid">
      <label>Ticker / nome<input type="text" name="ticker" required value="${esc(a.ticker)}"></label>
      <label>Classe<select name="cls">${Object.keys(ASSET_CLASSES).map(c => `<option ${c === a.class ? 'selected' : ''}>${c}</option>`).join('')}</select></label>
      ${manual ? `<label>Atualização do valor<select name="indexer">${Object.entries(INDEXERS).map(([k, l]) => `<option value="${k}" ${(a.indexer || '') === k ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
        <label>Taxa (ex.: 110 p/ 110% CDI, 6 p/ IPCA+6%)<input type="number" step="any" name="idxRate" value="${a.idxRate ?? ''}"></label>
        <label class="full">Valor atual total (R$) — só se for manual<input type="number" step="any" name="manualValue" value="${a.manualValue ?? ''}"></label>
        <p class="full muted small">Com indexador, o valor é atualizado sozinho (estimativa bruta, antes do IR) usando CDI/IPCA do Banco Central.</p>
        ${a.class === 'Tesouro Direto' ? `<div class="field full"><strong class="small" style="color:var(--text)">Preço de mercado real (marcação a mercado):</strong></div>
        <label>Código do título<input type="text" name="tdSymbol" value="${esc(a.tdSymbol || '')}" placeholder="tesouro-ipca-15052035"></label>
        <label>Quantidade de títulos (ex.: 2,35)<input type="number" step="any" name="tdQty" value="${a.tdQty ?? ''}"></label>
        <p class="full muted small">Com o código e a quantidade (veja no extrato da corretora), o valor usa o preço unitário do dia do Tesouro, buscado automaticamente — inclui a oscilação do IPCA+ e do prefixado. Precisa do token da brapi.dev.${a.tdPriceAt ? ` Último PU: ${fmt(a.tdPrice)} em ${new Date(a.tdPriceAt).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' })}.` : ''}</p>` : ''}` :
        `<label>Preço atual (R$)<input type="number" step="any" name="currentPrice" value="${a.currentPrice ?? ''}"></label>
         <label>ID CoinGecko (cripto)<input type="text" name="quoteId" value="${esc(a.quoteId || '')}" placeholder="bitcoin"></label>`}
    </div>`, d => {
      a.ticker = d.ticker.trim().toUpperCase(); a.class = d.cls;
      if ('manualValue' in d) a.manualValue = d.manualValue === '' ? null : +d.manualValue;
      if ('indexer' in d) { a.indexer = d.indexer; a.idxRate = d.idxRate === '' ? '' : +d.idxRate; }
      if ('tdSymbol' in d) { const sym = d.tdSymbol.trim().toLowerCase(); if (sym !== (a.tdSymbol || '')) { a.tdPrice = null; a.tdPriceAt = null; } a.tdSymbol = sym; a.tdQty = d.tdQty === '' ? '' : +d.tdQty; if (sym) setTimeout(() => autoQuotes(true), 100); }
      if ('currentPrice' in d) { a.currentPrice = d.currentPrice === '' ? null : +d.currentPrice; a.priceUpdatedAt = null; }
      if ('quoteId' in d) a.quoteId = d.quoteId.trim();
      recordSnapshot();
    });
  },
  'update-quotes': () => updateQuotes(),

  'pick-file': el => $('#' + el.dataset.inputId)?.click(),
  'cancel-import': () => { ui.pendingImport = null; render(); },
  'confirm-import': () => {
    const p = ui.pendingImport; const sel = p.items.filter(i => i.include);
    for (const i of sel) state.transactions.push({ id: uid(), date: i.date, description: i.description, amount: i.amount, tag: i.tag, accountId: p.accountId, source: 'importado', externalId: i.externalId });
    ui.pendingImport = null; save(); toast(`${sel.length} transações importadas.`);
    ui.view = 'transactions'; ui.filters = { month: '__all', q: '', tag: '__all', account: p.accountId, day: '' }; render();
  },
  'cancel-inv-import': () => { ui.pendingInvestImport = null; render(); },
  'confirm-inv-import': () => {
    const rows = ui.pendingInvestImport.rows.filter(r => !r.duplicate);
    for (const r of rows) { const a = findOrCreateAsset(r.ticker); state.investments.operations.push({ id: uid(), assetId: a.id, date: r.date, type: r.type, qty: r.qty, price: r.price, fees: r.fees }); }
    ui.pendingInvestImport = null; recordSnapshot(); save(); toast(`${rows.length} operações importadas.`);
    ui.view = 'investments'; render();
  },

  export: () => {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `financas-backup-${todayISO()}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  },
  reset: () => { if (confirm('Apagar TODOS os dados? Exporte um backup antes.' + (Sync.enabled() ? '\n\nA sincronização está ativa: os dados também serão apagados na nuvem e nos outros aparelhos.' : ''))) { state = defaultState(); save(); render(); } },
  'modal-cancel': () => { ui.pendingSync = null; $('#modal').close(); },
  'clear-logo': () => {},
  'open-sim': el => { const d = state.debts.find(x => x.id === el.dataset.id); ui.cons = null; ui.sim = { debtId: d.id, items: JSON.parse(JSON.stringify(d.plan || [])) }; ui.view = 'debts'; render(); window.scrollTo(0, 0); fetchMarket(); },
  'open-analysis': () => { ui.analysis = true; ui.view = 'investments'; render(); window.scrollTo(0, 0); fetchMarket(); },
  'close-analysis': () => { ui.analysis = false; render(); window.scrollTo(0, 0); },
  'trade-pick': el => { ui.trade = { id: el.dataset.id }; render(); $('#trade')?.scrollIntoView({ behavior: 'smooth' }); },
  'trade-register': () => {
    const { list } = portfolio(); const tr = ui.trade || {};
    const sp = list.find(p => p.asset.id === tr.id && p.qty > 1e-9) || list.find(p => p.qty > 1e-9 && !ASSET_CLASSES[p.asset.class]?.manual);
    if (!sp) return;
    const qty = Math.min(+tr.qty || sp.qty, sp.qty), price = +tr.price || sp.price;
    if (!confirm(`Registrar venda de ${fmtNum(qty)} ${sp.asset.ticker} a ${fmt(price)}?`)) return;
    state.investments.operations.push({ id: uid(), assetId: sp.asset.id, date: todayISO(), type: 'venda', qty, price, fees: 0 });
    recordSnapshot(); save(); ui.trade = null; render(); toast('Venda registrada na carteira.');
  },
  'open-cons': el => { ui.sim = null; ui.cons = el.dataset.id; ui.view = 'debts'; render(); window.scrollTo(0, 0); fetchMarket(); },
  'close-cons': () => { ui.cons = null; render(); window.scrollTo(0, 0); },
  'market-refresh': () => { fetchMarket(true).then(() => toast(market.cdi ? `CDI atualizado: ${fmtPctNum(market.cdi)} a.a.` : 'Não consegui buscar o CDI agora; informe manualmente.')); },
  'fs-add': el => {
    const d = state.debts.find(x => x.id === ui.sim.debtId); const st = financingStatus(d);
    ui.sim.items.push({ value: +el.dataset.value, every: +el.dataset.every || 0, start: el.dataset.start || (st.next ? st.next.ym : currentMonth()), end: '', mode: 'prazo' });
    render(); toast('Adicionado ao plano — veja o resultado no topo. Lembre de salvar.');
  },
  'fs-use': el => {
    const d = state.debts.find(x => x.id === ui.sim.debtId); const st = financingStatus(d);
    const sc = strategyScenarios(d, st.next.ym).list.find(x => x.key === el.dataset.scn);
    if (sc) { ui.sim.items = JSON.parse(JSON.stringify(sc.items)); render(); window.scrollTo(0, 0); toast('Estratégia aplicada ao plano. Lembre de salvar.'); }
  },
  'close-sim': () => { ui.sim = null; render(); window.scrollTo(0, 0); },
  'sim-preset': el => { $('#sp-value').value = el.dataset.value; $('#sp-every').value = el.dataset.every; $('#sp-start').value = el.dataset.start; $('#sp-end').value = ''; $('#sp-value').focus(); $('#sp-value').select(); },
  'sim-add': () => {
    const value = +$('#sp-value').value;
    if (!(value > 0)) return toast('Informe o valor da amortização.');
    ui.sim.items.push({ value, every: +$('#sp-every').value, start: $('#sp-start').value || currentMonth(), end: $('#sp-end').value || '', mode: $('#sp-mode').value });
    render();
  },
  'sim-del': el => { ui.sim.items.splice(+el.dataset.idx, 1); render(); },
  'sim-clear': () => { ui.sim.items = []; render(); },
  'sim-save': () => { const d = state.debts.find(x => x.id === ui.sim.debtId); d.plan = JSON.parse(JSON.stringify(ui.sim.items)); save(); render(); toast('Plano salvo.'); },
  'more-menu': () => openModal('Mais', `<div class="quick quick-grid">
      <button class="qa" data-action="nav" data-view="debts"><span class="qa-i">🏡</span>Financiamentos e consórcios</button>
      <button class="qa" data-action="nav" data-view="import"><span class="qa-i">⬆</span>Importar extrato</button>
      <button class="qa" data-action="nav" data-view="settings"><span class="qa-i">⚙</span>Configurações</button></div>`, null, { noSubmit: true }),
  'add-debt': el => { const kind = el.dataset.kind; openModal(kind === 'consorcio' ? 'Novo consórcio' : 'Novo financiamento', debtForm({}, kind), d => saveDebt(null, kind, d)); },
  'edit-debt': el => { const x = state.debts.find(d => d.id === el.dataset.id); openModal(`Editar ${x.name}`, debtForm(x, x.kind), d => saveDebt(x, x.kind, d)); },
  'del-debt': el => {
    const x = state.debts.find(d => d.id === el.dataset.id);
    if (!confirm(`Excluir "${x.name}"? As transações já lançadas continuam.`)) return;
    state.debts = state.debts.filter(d => d !== x); state.recurring = state.recurring.filter(r => r.debtId !== x.id); save(); render();
  },
  'add-extra': el => {
    const x = state.debts.find(d => d.id === el.dataset.id);
    openModal('Registrar amortização realizada', extraForm(), f => { const e = extraFromForm(f); if (!e) return false; (x.extras = x.extras || []).push(e); toast('Amortização registrada.'); });
  },
  'edit-extra': el => {
    const x = state.debts.find(d => d.id === el.dataset.id); const i = +el.dataset.idx;
    openModal('Atualizar amortização', extraForm(x.extras[i]), f => { const e = extraFromForm(f); if (!e) return false; x.extras[i] = e; });
  },
  'sim-realize': el => {
    const x = state.debts.find(d => d.id === ui.sim.debtId); const idx = +el.dataset.idx; const it = ui.sim.items[idx];
    const st = financingStatus(x); const occ = expandPlan([it], st.next ? st.next.ym : currentMonth(), '9999-12')[0];
    openModal('Concretizar amortização', extraForm({ date: todayISO(), value: it.value, mode: it.mode }), f => {
      const e = extraFromForm(f); if (!e) return false;
      (x.extras = x.extras || []).push(e);
      if (+it.every && occ) it.start = addMonths(occ.date.slice(0, 7), +it.every); else ui.sim.items.splice(idx, 1);
      x.plan = JSON.parse(JSON.stringify(ui.sim.items));
      toast('Amortização registrada! Quando o banco atualizar, informe os novos números em ✎.', 7000);
    }, { submitLabel: 'Registrar como realizada' });
  },
  'del-extra': el => { const x = state.debts.find(d => d.id === el.dataset.id); x.extras.splice(+el.dataset.idx, 1); save(); render(); },
  'anchor-debt': el => {
    const x = state.debts.find(d => d.id === el.dataset.id); const st = financingStatus(x);
    if (!st.next) return toast('Financiamento já quitado.');
    const cur = x.anchor && x.anchor.ym === st.next.ym ? x.anchor : {};
    openModal('Atualizar pelos dados do banco', `<p class="muted small">Abra o app do banco e copie os números <strong>antes da parcela de ${monthLabel(st.next.ym)}</strong>. A tabela passa a ser calculada a partir deles (inclui TR, correções e amortizações que o banco já aplicou).</p>
      <div class="form-grid">
        <label class="full">Saldo devedor hoje (R$)<input type="number" step="0.01" min="0" name="balance" required value="${cur.balance ?? round2(st.saldo)}"></label>
        <label>Valor da próxima parcela (R$, opcional)<input type="number" step="0.01" min="0" name="installment" value="${cur.installment ?? ''}"></label>
        <label>Parcelas restantes (opcional)<input type="number" step="1" min="0" name="remaining" value="${cur.remaining ?? ''}"></label>
      </div>`,
      f => { x.anchor = { ym: st.next.ym, balance: +f.balance, installment: f.installment === '' ? '' : +f.installment, remaining: f.remaining === '' ? '' : +f.remaining }; toast('Financiamento atualizado com os dados do banco.'); });
  },
  'pick-icon': el => {
    const f = el.closest('form'); f.icon.value = el.dataset.icon;
    f.querySelectorAll('.icon-grid .sel').forEach(b => b.classList.remove('sel')); el.classList.add('sel');
  },

  'sync-connect': async el => {
    const token = $('#sync-token').value.trim(), password = $('#sync-pass').value;
    if (!token) return toast('Cole o token do GitHub.');
    if (password.length < 8) return toast('A senha precisa ter pelo menos 8 caracteres.');
    el.disabled = true; el.textContent = 'Conectando…';
    try {
      const res = await Sync.connect(token, password);
      if (res.found && hasData(state)) {
        ui.pendingSync = res;
        openModal('Já existem dados na nuvem', `<p>Encontrei dados sincronizados (última alteração em ${new Date(res.remote.updatedAt).toLocaleString('pt-BR')}, pelo ${esc(res.remote.device || 'outro aparelho')}). Este aparelho também tem dados. Qual versão você quer manter?</p>
          <div class="quick quick-grid">
            <button class="qa" data-action="sync-choose" data-choice="pull"><span class="qa-i">⬇</span>Usar os dados da nuvem</button>
            <button class="qa" data-action="sync-choose" data-choice="push"><span class="qa-i">⬆</span>Enviar os deste aparelho</button>
          </div><p class="muted small">A versão descartada fica guardada como backup em Configurações › Backup.</p>`, null, { noSubmit: true });
        return;
      }
      if (res.found) { Sync.adopt(res); toast('Conectado! Dados baixados da nuvem.'); }
      else {
        if (!state.meta?.updatedAt) { state.meta = { ...(state.meta || {}), updatedAt: Date.now() }; persist(); }
        await Sync.pushLocal(res); toast('Conectado! Seus dados foram enviados (criptografados) para a nuvem.');
      }
      render();
    } catch (e) { toast(e.message, 8000); if (el.isConnected) { el.disabled = false; el.textContent = 'Conectar'; } }
  },
  'sync-choose': async el => {
    const res = ui.pendingSync; ui.pendingSync = null; $('#modal').close();
    if (!res) return;
    try {
      if (el.dataset.choice === 'pull') {
        try { localStorage.setItem('minhas-financas.backup-conflito', JSON.stringify({ at: Date.now(), origin: 'deste aparelho antes de conectar', state })); } catch { /* */ }
        Sync.adopt(res); toast('Sincronização ativada com os dados da nuvem.');
      } else {
        state.meta = { ...(state.meta || {}), updatedAt: Date.now() }; persist();
        await Sync.pushLocal(res); toast('Sincronização ativada: dados deste aparelho enviados.');
      }
      render();
    } catch (e) { toast(e.message, 8000); }
  },
  'sync-now': () => Sync.syncNow().then(() => { if (ui.view === 'settings') render(); }),
  'sync-disconnect': () => {
    if (!confirm('Desconectar este aparelho da sincronização? Os dados continuam aqui e a nuvem não é apagada.')) return;
    Sync.disconnect(); render();
  },
  'restore-conflict': () => {
    const c = Sync.conflictBackup(); if (!c || !confirm('Substituir os dados atuais pelo backup de conflito?')) return;
    state = normalizeState(c.state); Sync.clearConflictBackup(); save(); render(); toast('Backup restaurado.');
  },
  'clear-conflict': () => { Sync.clearConflictBackup(); render(); },
};

function wireAccountForm(dlg) {
  const f = dlg.querySelector('form');
  const preview = () => { $('#acc-preview').innerHTML = avatar({ name: f.name.value, bank: f.bank.value, color: f.color.value, type: f.type.value, logo: f.logo.value }); };
  f.addEventListener('input', preview);
  f.addEventListener('change', async e => {
    if (e.target.name !== 'logoFile') return preview();
    const file = e.target.files[0]; if (!file) return;
    try { f.logo.value = await resizeImage(file); preview(); } catch (err) { toast(err.message); }
  });
  f.addEventListener('click', e => { if (e.target.closest('[data-action=clear-logo]')) { f.logo.value = ''; preview(); } });
}
function accountForm(a = {}) {
  return `<div class="form-grid">
    <label class="full">Nome<input type="text" name="name" required value="${esc(a.name || '')}" placeholder="Nubank, Itaú, Cartão Inter…"></label>
    <label>Tipo<select name="type">
      <option value="corrente" ${a.type === 'corrente' ? 'selected' : ''}>Conta corrente / poupança</option>
      <option value="cartao" ${a.type === 'cartao' ? 'selected' : ''}>Cartão de crédito</option>
      <option value="corretora" ${a.type === 'corretora' ? 'selected' : ''}>Corretora</option></select></label>
    <label>Saldo inicial (R$)<input type="number" step="0.01" name="initialBalance" value="${a.initialBalance ?? 0}"></label>
    <label>Limite do cartão (R$)<input type="number" step="0.01" name="limit" value="${a.limit || ''}" placeholder="só para cartões"></label>
    <label>Banco (ícone)<select name="bank">
      <option value="" ${!a.bank ? 'selected' : ''}>Automático pelo nome</option>
      ${BANKS.map(b => `<option value="${b.id}" ${a.bank === b.id ? 'selected' : ''}>${esc(b.name)}</option>`).join('')}
      <option value="none" ${a.bank === 'none' ? 'selected' : ''}>Nenhum (iniciais + cor)</option></select></label>
    <label>Cor (sem banco)<input type="color" name="color" value="${a.color || accountColor(a)}" style="height:38px"></label>
    <div class="field full">Imagem própria (opcional)
      <div class="acc-preview"><span id="acc-preview">${avatar(a)}</span>
        <input type="file" name="logoFile" accept="image/*">
        <button type="button" class="btn small ghost" data-action="clear-logo">Remover imagem</button></div>
      <input type="hidden" name="logo" value="${esc(a.logo || '')}">
    </div>
  </div><p class="muted small">"Saldo em conta" soma apenas contas corrente/poupança: saldo inicial + transações até o fim do mês.</p>`;
}
function tagForm(t = {}) {
  return `<div class="form-grid">
    <label>Nome<input type="text" name="name" required value="${esc(t.name || '')}"></label>
    <label>Cor<input type="color" name="color" value="${t.color || '#2563eb'}" style="height:38px"></label>
    <div class="field full">Ícone
      <input type="text" name="icon" value="${esc(t.icon || '🏷️')}" maxlength="8" class="icon-input">
      <div class="icon-grid">${ICON_CHOICES.map(i => `<button type="button" data-action="pick-icon" data-icon="${i}" class="${i === t.icon ? 'sel' : ''}">${i}</button>`).join('')}</div>
    </div>
    <label class="full">Grupo (equilíbrio 50/30/20)<select name="group">${[['', 'Nenhum (receitas, transferências)'], ['essencial', 'Essencial — até 50% da receita'], ['livre', 'Não essencial — até 30%'], ['investimento', 'Investimento — mínimo 20%']].map(([v, l]) => `<option value="${v}" ${(t.group || '') === v ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
    <label class="check full"><input type="checkbox" name="exclude" ${t.exclude ? 'checked' : ''}> Não contar como receita/despesa (ex.: transferências entre contas, pagamento de fatura)</label>
  </div>`;
}
function recForm(r = {}) {
  return `<div class="form-grid">
    <label class="full">Descrição<input type="text" name="description" required value="${esc(r.description || '')}"></label>
    <label>Valor (negativo = despesa)<input type="number" step="0.01" name="amount" required value="${r.amount ?? ''}"></label>
    <label>Dia do mês<input type="number" min="1" max="31" name="day" required value="${r.day || 5}"></label>
    <label>Tag<select name="tag">${tagOptions(r.tag || '')}</select></label>
    <label>Conta<select name="accountId">${accountOptions(r.accountId || state.accounts[0]?.id)}</select></label>
    <label>A partir de<input type="month" name="startMonth" required value="${r.startMonth || currentMonth()}"></label>
    <label>Até (opcional)<input type="month" name="endMonth" value="${r.endMonth || ''}"></label>
  </div><p class="muted small">O lançamento é criado automaticamente quando a data chega (ao abrir o app). Se o valor também vem no extrato importado, a importação marca como possível duplicada.</p>`;
}
const recFromForm = d => ({ description: d.description.trim(), amount: round2(+d.amount), day: +d.day, tag: d.tag, accountId: d.accountId, startMonth: d.startMonth, endMonth: d.endMonth || '' });

/* ================================================================ events */
document.addEventListener('click', e => {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const fn = actions[el.dataset.action];
  if (fn) { e.preventDefault(); fn(el, e); }
});

document.addEventListener('change', e => {
  const el = e.target;
  if (el.type === 'file') {
    const files = [...el.files]; el.value = '';
    if (!files.length) return;
    if (el.id === 'file-bank') handleBankFiles(files);
    else if (el.id === 'file-invest') handleInvestFile(files[0]);
    else if (el.id === 'file-backup') files[0].text().then(t => {
      try { const s = JSON.parse(t); if (!s.transactions) throw new Error('formato inválido'); state = normalizeState(s); save(); render(); toast('Backup restaurado.'); }
      catch (err) { toast('Erro ao ler backup: ' + err.message); }
    });
    return;
  }
  const k = el.dataset.change; if (!k) return;
  if (k === 'tx-tag') { const t = state.transactions.find(x => x.id === el.dataset.id); t.tag = el.value; save(); if (ui.view === 'transactions') render(); }
  else if (k === 'sim-fin-mode') { const id = el.dataset.id; SIM[id] = { ...SIM[id], mode: el.value }; $('#sim-' + id).innerHTML = simText(state.debts.find(d => d.id === id)); }
  else if (k === 'fs' || k === 'cs') {
    const d = state.debts.find(x => x.id === el.dataset.id);
    d.strategy = { ...(d.strategy || {}), [el.dataset.field]: el.type === 'checkbox' ? el.checked : el.value };
    save(); render();
  }
  else if (k === 'pa') {
    const f = el.dataset.field;
    if (f === 'cdi') state.settings.cdi = el.value;
    else { const c = f.slice(2); state.investments.targets = { ...(state.investments.targets || {}), [c]: el.value === '' ? 0 : +el.value }; }
    save(); render();
  }
  else if (k === 'pa-aporte') { ui.aporte = +el.value || 0; render(); }
  else if (k === 'tr') { ui.trade = { ...(ui.trade || {}), [el.dataset.field]: el.value }; if (el.dataset.field === 'id') { ui.trade.qty = ''; ui.trade.price = ''; } render(); $('#trade')?.scrollIntoView(); }
  else if (k === 'filter-month') { ui.filters.month = el.value; render(); }
  else if (k === 'filter-tag') { ui.filters.tag = el.value; render(); }
  else if (k === 'filter-account') { ui.filters.account = el.value; render(); }
  else if (k === 'imp-include') { ui.pendingImport.items[+el.dataset.idx].include = el.checked; render(); }
  else if (k === 'imp-tag') { ui.pendingImport.items[+el.dataset.idx].tag = el.value; }
});

let inputTimer;
document.addEventListener('input', e => {
  const k = e.target.dataset.input; if (!k) return;
  if (k === 'filter-q') {
    ui.filters.q = e.target.value;
    clearTimeout(inputTimer);
    inputTimer = setTimeout(() => { const pos = e.target.selectionStart; render(); const i = $('[data-input=filter-q]'); i.focus(); i.setSelectionRange(pos, pos); }, 250);
  }
  if (k === 'sim-fin') { const id = e.target.dataset.id; SIM[id] = { ...SIM[id], value: e.target.value }; $('#sim-' + id).innerHTML = simText(state.debts.find(d => d.id === id)); }
  if (k === 'sim-cons') { const id = e.target.dataset.id; SIM[id] = { ...SIM[id], [e.target.dataset.field]: e.target.value }; $('#csim-' + id).innerHTML = consSimText(state.debts.find(d => d.id === id)); }
  if (k === 'set-name') { state.settings.name = e.target.value; save(); }
  if (k === 'set-token') { state.settings.brapiToken = e.target.value; save(); }
});

['dragover', 'dragleave', 'drop'].forEach(ev => document.addEventListener(ev, e => {
  const z = e.target.closest?.('[data-drop]');
  if (!z) return;
  e.preventDefault();
  z.classList.toggle('over', ev === 'dragover');
  if (ev === 'drop') {
    const files = [...e.dataTransfer.files];
    if (z.dataset.drop === 'bank') handleBankFiles(files); else handleInvestFile(files[0]);
  }
}));

/* ================================================================== boot */
Sync.init({
  getState: () => state,
  applyRemote: data => { state = normalizeState(data); applyQuoteCache(); persist(); safeRender(); },
  onStatus: renderSyncStatus,
});
render();
(async () => {
  // Baixa as novidades da nuvem antes de gerar os recorrentes, para os aparelhos não brigarem.
  if (Sync.enabled()) await Sync.syncNow();
  const generated = runRecurring();
  if (generated) { render(); toast(`${generated} lançamento(s) recorrente(s) gerado(s) automaticamente.`); }
  // Atualizações automáticas: indicadores do BC e cotações ao abrir, a cada 5 min (respeitando o intervalo) e ao voltar ao app
  await fetchMarket();
  await autoQuotes();
  setInterval(() => { if (document.visibilityState === 'visible') { autoQuotes(); fetchMarket(); } }, 5 * 60e3);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') { autoQuotes(); fetchMarket(); } });
})();
