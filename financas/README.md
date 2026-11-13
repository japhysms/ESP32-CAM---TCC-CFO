# 💰 Minhas Finanças

Sistema pessoal para acompanhar todo o fluxo de dinheiro (contas, cartão e investimentos), com o máximo de automação possível.
Roda 100% no navegador, sem servidor e sem instalação: **abra o `index.html`**.

## O que tem

**Visão geral** (painel principal)
- Resultado do período, receitas, despesas e saldo em conta, com navegação entre meses
- *Saídas por dia*: calendário de calor mostrando quanto saiu em cada dia (clique num dia para ver as transações)
- *Por tags*: rosca de despesas/receitas por categoria (clique numa tag para filtrar)
- Receitas × despesas dos últimos 6 meses, resumo dos investimentos e maiores gastos do mês

**Investimentos**
- Carteira consolidada: preço médio, custo, valor atual, resultado (R$ e %) e peso de cada ativo
- Classes: Ações, FIIs, ETFs, BDRs, Renda Fixa, Tesouro Direto, Cripto, Exterior
- Compras, vendas (com lucro realizado) e proventos (dividendos, JCP, rendimentos)
- Alocação por classe, aportes por mês e evolução do patrimônio
- **Cotações automáticas**: B3 via [brapi.dev](https://brapi.dev) (token gratuito em Configurações) e cripto via CoinGecko
- Renda fixa/Tesouro: informe o valor atual de vez em quando (✎)

**Automação**
- **Importação de extratos** OFX (quase todos os bancos) e CSV (Nubank conta e cartão, Inter, Itaú, C6…), com detecção automática de colunas, formato de número brasileiro e encoding
- **Deduplicação**: importar o mesmo extrato duas vezes não duplica nada; lançamentos manuais/recorrentes parecidos aparecem como "possível duplicada"
- **Regras de categorização**: "se a descrição contém `ifood|rappi` → Alimentação". Já vem com regras para os gastos mais comuns no Brasil; crie novas com ⚡ direto numa transação
- **Lançamentos recorrentes** (salário, aluguel, assinaturas) gerados sozinhos todo mês
- **Importação de operações** a partir do CSV de Negociação da Área do Investidor B3 ou planilha da corretora
- Tag "Transferência" fica fora de receitas/despesas (pagamento de fatura, transferência entre contas suas)

## Como usar

1. Abra `index.html` no navegador (ou clique em *Carregar dados de exemplo* para conhecer).
2. Em **Configurações**, cadastre suas contas (conta corrente, cartão, corretora) e seu nome.
3. Em **Importar**, arraste o OFX/CSV exportado do app do seu banco. Confira as tags sugeridas e confirme.
4. Em **Investimentos**, registre operações ou importe o CSV da B3, e clique em *Atualizar cotações*.
5. Repita a importação 1× por semana/mês — duplicatas são ignoradas automaticamente.

Arquivos de teste em [`exemplos/`](exemplos/).

## Dados e privacidade

Os dados ficam apenas no `localStorage` do seu navegador — nada é enviado para servidores (exceto os tickers, ao buscar cotações).
Use **Configurações › Backup › Exportar JSON** com frequência e **Restaurar JSON** para levar para outro computador.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `index.html` | estrutura da página |
| `styles.css` | visual (claro/escuro automático, responsivo) |
| `app.js` | estado, cálculos, telas, regras, recorrências, carteira e cotações |
| `importers.js` | leitores de OFX, CSV bancário e CSV de operações |
