/* Sincronização entre aparelhos via GitHub Gist secreto.
 * Os dados são comprimidos e criptografados no próprio aparelho (PBKDF2 + AES-256-GCM)
 * com a senha do usuário antes do envio — o GitHub só guarda texto cifrado.
 * Estratégia: o estado inteiro é um documento; vence a versão alterada por último
 * (a versão descartada num conflito fica salva como backup local). */
'use strict';

const Sync = (() => {
  const CFG_KEY = 'minhas-financas.sync';
  const CONFLICT_KEY = 'minhas-financas.backup-conflito';
  const FILE = 'minhas-financas.json';
  const DESC = 'Minhas Finanças — dados sincronizados (criptografados)';
  const API = 'https://api.github.com';
  const ITERATIONS = 250000;

  let cfg = readCfg();
  let status = { state: enabled() ? 'ok' : 'off', msg: '' };
  let hooks = { getState: () => ({}), applyRemote: () => {}, onStatus: () => {} };
  let pushTimer = null;
  let running = null;
  let keyCache = null;

  function readCfg() { try { return JSON.parse(localStorage.getItem(CFG_KEY)) || {}; } catch { return {}; } }
  function saveCfg() { try { localStorage.setItem(CFG_KEY, JSON.stringify(cfg)); } catch { /* sem storage */ } }
  function enabled() { return !!(cfg.token && cfg.password && cfg.gistId); }
  function setStatus(state, msg = '') { status = { state, msg }; hooks.onStatus(status); }
  const deviceName = () => (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) ? 'celular' : 'computador');

  /* ---------------------------------------------------------- criptografia */
  const toB64 = buf => {
    const b = new Uint8Array(buf); let s = '';
    for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const fromB64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const randomSalt = () => crypto.getRandomValues(new Uint8Array(16));

  async function deriveKey(password, salt) {
    const id = password + '|' + toB64(salt);
    if (keyCache && keyCache.id === id) return keyCache.key;
    const base = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey({ name: 'PBKDF2', salt, iterations: ITERATIONS, hash: 'SHA-256' },
      base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
    keyCache = { id, key };
    return key;
  }
  async function pipe(bytes, stream) {
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());
  }
  async function encrypt(obj, password, salt) {
    let bytes = new TextEncoder().encode(JSON.stringify(obj));
    const gz = typeof CompressionStream !== 'undefined';
    if (gz) bytes = await pipe(bytes, new CompressionStream('gzip'));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await deriveKey(password, salt), bytes);
    return { salt: toB64(salt), iv: toB64(iv), gz, data: toB64(ct) };
  }
  async function decrypt(payload, password) {
    let bytes;
    try {
      const key = await deriveKey(password, fromB64(payload.salt));
      bytes = new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: fromB64(payload.iv) }, key, fromB64(payload.data)));
    } catch { throw new Error('Senha de sincronização incorreta (não consegui abrir os dados da nuvem).'); }
    if (payload.gz) bytes = await pipe(bytes, new DecompressionStream('gzip'));
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  /* ---------------------------------------------------------------- GitHub */
  async function gh(path, opts = {}) {
    let r;
    try {
      r = await fetch(API + path, {
        ...opts,
        headers: {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${cfg.token}`,
          'X-GitHub-Api-Version': '2022-11-28',
          ...(opts.body ? { 'Content-Type': 'application/json' } : {}),
        },
      });
    } catch { throw new Error('Sem conexão com o GitHub. As alterações ficam salvas aqui e serão enviadas depois.'); }
    if (r.status === 401) throw new Error('Token do GitHub inválido ou expirado. Gere um novo e conecte de novo.');
    if (r.status === 403) throw new Error('O GitHub recusou o acesso: confira se o token tem a permissão "gist" (ou aguarde, se for limite de uso).');
    if (r.status === 404) throw new Error('Gist de sincronização não encontrado (foi apagado?). Desconecte e conecte de novo.');
    if (!r.ok) { let m = ''; try { m = (await r.json()).message; } catch { /* */ } throw new Error(`Erro do GitHub ${r.status}${m ? ': ' + m : ''}`); }
    return r.json();
  }
  async function findGist() {
    for (let page = 1; page <= 10; page++) {
      const list = await gh(`/gists?per_page=100&page=${page}`);
      const g = list.find(x => x.description === DESC && x.files && x.files[FILE]);
      if (g) return g.id;
      if (list.length < 100) break;
    }
    return null;
  }
  async function readRemote() {
    const g = await gh(`/gists/${cfg.gistId}`);
    const f = g.files && g.files[FILE];
    if (!f) return null;
    let content = f.content;
    if (f.truncated) content = await (await fetch(f.raw_url)).text();
    const payload = JSON.parse(content);
    if (payload.app !== 'minhas-financas') throw new Error('O Gist encontrado não é do Minhas Finanças.');
    return payload;
  }
  async function writeRemote(state, salt) {
    const payload = {
      app: 'minhas-financas', v: 1,
      updatedAt: (state.meta && state.meta.updatedAt) || Date.now(),
      device: deviceName(),
      ...(await encrypt(state, cfg.password, salt)),
    };
    const files = { [FILE]: { content: JSON.stringify(payload) } };
    if (cfg.gistId) await gh(`/gists/${cfg.gistId}`, { method: 'PATCH', body: JSON.stringify({ description: DESC, files }) });
    else cfg.gistId = (await gh('/gists', { method: 'POST', body: JSON.stringify({ description: DESC, public: false, files }) })).id;
    cfg.salt = payload.salt; cfg.lastSynced = payload.updatedAt; cfg.lastSyncAt = Date.now();
    saveCfg();
  }

  function keepConflictBackup(obj, origin) {
    try { localStorage.setItem(CONFLICT_KEY, JSON.stringify({ at: Date.now(), origin, state: obj })); } catch { /* */ }
  }

  /* --------------------------------------------------------------- público */
  function syncNow() {
    if (!enabled()) return Promise.resolve();
    if (running) return running;
    clearTimeout(pushTimer);
    running = (async () => {
      setStatus('syncing');
      try {
        const local = hooks.getState();
        const L = (local.meta && local.meta.updatedAt) || 0;
        const base = cfg.lastSynced || 0;
        const remote = await readRemote();
        const R = remote ? remote.updatedAt || 0 : -1;
        let msg = '';
        if (!remote) await writeRemote(local, cfg.salt ? fromB64(cfg.salt) : randomSalt());
        else if (R > L) {
          const data = await decrypt(remote, cfg.password);
          if (L > base) { keepConflictBackup(local, 'este aparelho'); msg = 'Conflito: os dois aparelhos mudaram; mantive a versão mais recente (a deste aparelho ficou como backup).'; }
          hooks.applyRemote(data);
          cfg.salt = remote.salt; cfg.lastSynced = R; cfg.lastSyncAt = Date.now(); saveCfg();
        } else if (L > R) {
          if (R > base) { keepConflictBackup(await decrypt(remote, cfg.password), 'nuvem'); msg = 'Conflito: os dois aparelhos mudaram; mantive a versão mais recente (a da nuvem ficou como backup).'; }
          await writeRemote(local, fromB64(remote.salt));
        } else { cfg.lastSynced = R; cfg.lastSyncAt = Date.now(); saveCfg(); }
        setStatus('ok', msg);
      } catch (e) {
        setStatus('error', e.message);
      } finally { running = null; }
    })();
    return running;
  }

  /** Chamado a cada alteração local: envia alguns segundos depois (agrupa várias edições). */
  function schedulePush() {
    if (!enabled()) return;
    setStatus('pending');
    clearTimeout(pushTimer);
    pushTimer = setTimeout(syncNow, 2500);
  }

  /** Testa token e senha. Não ativa nada ainda: devolve o que achou para o app decidir. */
  async function connect(token, password) {
    const prev = cfg;
    cfg = { token, password };
    try {
      const gistId = await findGist();
      if (!gistId) return { found: false, cfg };
      cfg.gistId = gistId;
      const remote = await readRemote();
      if (!remote) return { found: false, cfg };
      const data = await decrypt(remote, password);
      return { found: true, cfg, remote, data };
    } finally { cfg = prev; }
  }
  /** Ativa usando os dados da nuvem. */
  function adopt(res) {
    cfg = { ...res.cfg, salt: res.remote.salt, lastSynced: res.remote.updatedAt, lastSyncAt: Date.now() };
    saveCfg();
    hooks.applyRemote(res.data);
    setStatus('ok');
  }
  /** Ativa enviando os dados deste aparelho (substitui a nuvem, se houver). */
  async function pushLocal(res) {
    const prev = cfg;
    cfg = { ...res.cfg };
    try {
      setStatus('syncing');
      await writeRemote(hooks.getState(), res.remote ? fromB64(res.remote.salt) : randomSalt());
      setStatus('ok');
    } catch (e) { cfg = prev; setStatus(prev.gistId ? 'error' : 'off', e.message); throw e; }
  }
  function disconnect() { clearTimeout(pushTimer); cfg = {}; keyCache = null; saveCfg(); setStatus('off'); }

  function conflictBackup() { try { return JSON.parse(localStorage.getItem(CONFLICT_KEY)); } catch { return null; } }
  function clearConflictBackup() { try { localStorage.removeItem(CONFLICT_KEY); } catch { /* */ } }

  function init(h) {
    hooks = { ...hooks, ...h };
    hooks.onStatus(status);
    // Ao voltar para o app: baixa novidades. Ao sair (trocar de app no celular): envia o que estiver pendente.
    document.addEventListener('visibilitychange', () => {
      if (!enabled()) return;
      if (document.visibilityState === 'hidden') { if (status.state === 'pending') syncNow(); }
      else if (Date.now() - (cfg.lastSyncAt || 0) > 15000) syncNow();
    });
    window.addEventListener('online', () => { if (enabled()) syncNow(); });
    setInterval(() => { if (enabled() && document.visibilityState === 'visible') syncNow(); }, 120000);
  }

  return {
    init, syncNow, schedulePush, connect, adopt, pushLocal, disconnect, enabled,
    status: () => status,
    info: () => ({ gistId: cfg.gistId, lastSyncAt: cfg.lastSyncAt, lastSynced: cfg.lastSynced }),
    conflictBackup, clearConflictBackup,
  };
})();
