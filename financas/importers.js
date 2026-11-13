/* Importadores de extratos: OFX (padrão dos bancos brasileiros), CSV genérico
 * (Nubank, Inter, Itaú, C6, etc.) e CSV de operações de investimento (B3/corretoras). */
'use strict';

const Importers = (() => {
  const norm = s => String(s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().trim();
  const pad = n => String(n).padStart(2, '0');

  /** Converte "1.234,56", "-12.50", "R$ 10,00", "(5,00)" e "5,00-" em número. */
  function parseNumber(s) {
    if (typeof s === 'number') return s;
    s = String(s ?? '').trim().replace(/R\$\s?/i, '').replace(/\s/g, '');
    if (!s) return NaN;
    let neg = false;
    if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
    if (s.endsWith('-')) { neg = true; s = s.slice(0, -1); }
    if (s.startsWith('-')) { neg = true; s = s.slice(1); }
    if (s.startsWith('+')) s = s.slice(1);
    const lastComma = s.lastIndexOf(','), lastDot = s.lastIndexOf('.');
    if (lastComma > lastDot) s = s.replace(/\./g, '').replace(',', '.');
    else s = s.replace(/,/g, '');
    const n = parseFloat(s);
    if (isNaN(n)) return NaN;
    return neg ? -n : n;
  }

  /** Aceita dd/mm/aaaa, dd/mm/aa, dd-mm-aaaa, aaaa-mm-dd e aaaammdd. Retorna "aaaa-mm-dd" ou null. */
  function parseDate(s) {
    s = String(s ?? '').trim();
    let m;
    if ((m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/))) return `${m[1]}-${pad(m[2])}-${pad(m[3])}`;
    if ((m = s.match(/^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})/))) {
      const y = m[3].length === 2 ? '20' + m[3] : m[3];
      return `${y}-${pad(m[2])}-${pad(m[1])}`;
    }
    if ((m = s.match(/^(\d{4})(\d{2})(\d{2})/))) return `${m[1]}-${m[2]}-${m[3]}`;
    return null;
  }

  function decodeEntities(s) {
    return String(s).replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'");
  }

  /* ------------------------------------------------------------------ OFX */
  function parseOFX(text) {
    const isCard = /<CREDITCARDMSGSRSV1>|<CCSTMTRS>/i.test(text);
    const rows = [];
    const blocks = text.split(/<STMTTRN>/i).slice(1);
    for (const raw of blocks) {
      const body = raw.split(/<\/STMTTRN>/i)[0];
      const get = tag => {
        const m = body.match(new RegExp('<' + tag + '>([^<\\r\\n]*)', 'i'));
        return m ? decodeEntities(m[1].trim()) : '';
      };
      const date = parseDate(get('DTPOSTED'));
      const amount = parseNumber(get('TRNAMT').replace(',', '.'));
      if (!date || isNaN(amount)) continue;
      const name = get('NAME'), memo = get('MEMO');
      let description = memo || name;
      if (name && memo && !norm(memo).includes(norm(name))) description = `${name} - ${memo}`;
      rows.push({ date, amount, description: description || '(sem descrição)', id: get('FITID') || null });
    }
    let balance = null;
    const bal = text.match(/<LEDGERBAL>[\s\S]*?<BALAMT>([^<\r\n]+)/i);
    if (bal) balance = parseNumber(bal[1].replace(',', '.'));
    return { rows, isCard, balance };
  }

  /* ------------------------------------------------------------------ CSV */
  function detectDelimiter(line) {
    const counts = [';', ',', '\t', '|'].map(d => [d, splitLine(line, d).length]);
    counts.sort((a, b) => b[1] - a[1]);
    return counts[0][0];
  }

  function splitLine(line, delim) {
    const out = []; let cur = ''; let q = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (q) {
        if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
        else if (c === '"') q = false;
        else cur += c;
      } else if (c === '"') q = true;
      else if (c === delim) { out.push(cur); cur = ''; }
      else cur += c;
    }
    out.push(cur);
    return out.map(s => s.trim());
  }

  function readTable(text) {
    text = text.replace(/^﻿/, '');
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) return { header: [], data: [] };
    // Alguns bancos colocam linhas de cabeçalho/saldo antes da tabela: usa a primeira linha com >= 3 colunas.
    const delim = detectDelimiter(lines.find(l => detectDelimiter(l) && l.length > 5) || lines[0]);
    const table = lines.map(l => splitLine(l, delim));
    const width = Math.max(...table.map(r => r.length));
    const start = table.findIndex(r => r.length >= Math.min(3, width));
    return { table: table.slice(start < 0 ? 0 : start) };
  }

  function findCol(header, patterns, exclude = []) {
    const h = header.map(norm);
    for (const p of patterns) {
      const i = h.findIndex((c, idx) => !exclude.includes(idx) && p.test(c));
      if (i >= 0) return i;
    }
    return -1;
  }

  /** CSV de extrato bancário ou fatura. Retorna { rows, warnings }. */
  function parseBankCSV(text) {
    const { table } = readTable(text);
    if (!table || !table.length) return { rows: [], warnings: ['Arquivo vazio.'] };
    const warnings = [];
    let header = table[0];
    let body = table.slice(1);
    const headerLooksLikeData = header.some(c => parseDate(c));
    let iDate, iDesc, iAmount, iId, iCat, iCred, iDeb;

    if (headerLooksLikeData) {
      // Sem cabeçalho: deduz pelas primeiras linhas.
      body = table;
      const sample = table[0];
      iDate = sample.findIndex(c => parseDate(c));
      iAmount = sample.findIndex((c, i) => i !== iDate && !isNaN(parseNumber(c)) && /\d/.test(c));
      iDesc = sample.findIndex((c, i) => i !== iDate && i !== iAmount && c && isNaN(parseNumber(c)));
      iId = iCat = iCred = iDeb = -1;
      warnings.push('Arquivo sem cabeçalho: colunas deduzidas automaticamente.');
    } else {
      iDate = findCol(header, [/^data( (do )?lancamento)?$/, /^data/, /^date/, /^dt/]);
      iAmount = findCol(header, [/^valor( \(r\$\))?$/, /^valor/, /^amount/, /^quantia/, /^montante/]);
      iId = findCol(header, [/^identificador/, /^fitid/, /^id$/]);
      iDesc = findCol(header, [/^descri/, /^title/, /^titulo/, /^historico/, /^lancamento/, /^estabelecimento/, /^memo/, /^detalhe/], [iId, iDate]);
      iCat = findCol(header, [/^categoria/, /^category/]);
      iCred = findCol(header, [/^credito/, /^entrada/]);
      iDeb = findCol(header, [/^debito/, /^saida/]);
    }

    if (iDate < 0 || (iAmount < 0 && (iCred < 0 || iDeb < 0))) {
      return { rows: [], warnings: ['Não encontrei as colunas de data e valor. Cabeçalho lido: ' + header.join(' | ')] };
    }

    const rows = [];
    for (const r of body) {
      const date = parseDate(r[iDate]);
      if (!date) continue;
      let amount;
      if (iAmount >= 0) amount = parseNumber(r[iAmount]);
      else {
        const c = parseNumber(r[iCred]) || 0, d = parseNumber(r[iDeb]) || 0;
        amount = Math.abs(c) - Math.abs(d);
      }
      if (isNaN(amount) || amount === 0) continue;
      const description = (iDesc >= 0 ? r[iDesc] : '') || '(sem descrição)';
      rows.push({
        date, amount, description,
        id: iId >= 0 && r[iId] ? r[iId] : null,
        category: iCat >= 0 ? r[iCat] : '',
      });
    }
    return { rows, warnings };
  }

  /** CSV de operações de investimento (formato próprio ou exportação "Negociação" da Área do Investidor B3). */
  function parseInvestCSV(text) {
    const { table } = readTable(text);
    if (!table || table.length < 2) return { rows: [], warnings: ['Arquivo vazio.'] };
    const header = table[0];
    const iDate = findCol(header, [/^data/, /^date/]);
    const iTicker = findCol(header, [/^codigo/, /^ticker/, /^ativo/, /^papel/, /^produto/]);
    const iType = findCol(header, [/^tipo/, /^operacao/, /movimentacao/, /^c\/v/, /^side/]);
    const iQty = findCol(header, [/^quant/, /^qtd/]);
    const iPrice = findCol(header, [/^preco/, /^price/, /^valor unit/]);
    const iFees = findCol(header, [/^taxa/, /^custo/, /^fees/]);
    const iTotal = findCol(header, [/^valor( total)?$/, /^total/]);
    if (iDate < 0 || iTicker < 0 || (iPrice < 0 && iTotal < 0)) {
      return { rows: [], warnings: ['Colunas esperadas: data, ticker/código, tipo, quantidade, preço. Cabeçalho lido: ' + header.join(' | ')] };
    }
    const rows = [];
    for (const r of table.slice(1)) {
      const date = parseDate(r[iDate]);
      if (!date) continue;
      let ticker = String(r[iTicker] || '').toUpperCase().trim();
      ticker = ticker.replace(/F$/, m => (/^[A-Z]{4}\d{1,2}F$/.test(ticker) ? '' : m)); // mercado fracionário: PETR4F -> PETR4
      const t = norm(iType >= 0 ? r[iType] : 'compra');
      let type = 'compra';
      if (/^v|venda|sell/.test(t)) type = 'venda';
      else if (/provento|dividend|rendimento|jcp|juros/.test(t)) type = 'provento';
      const qty = iQty >= 0 ? Math.abs(parseNumber(r[iQty])) : 1;
      let price = iPrice >= 0 ? Math.abs(parseNumber(r[iPrice])) : NaN;
      if (isNaN(price) && iTotal >= 0) price = Math.abs(parseNumber(r[iTotal])) / (qty || 1);
      const fees = iFees >= 0 ? Math.abs(parseNumber(r[iFees])) || 0 : 0;
      if (!ticker || isNaN(price)) continue;
      rows.push({ date, ticker, type, qty: isNaN(qty) ? 1 : qty, price, fees });
    }
    return { rows, warnings: [] };
  }

  return { parseOFX, parseBankCSV, parseInvestCSV, parseNumber, parseDate, norm };
})();
