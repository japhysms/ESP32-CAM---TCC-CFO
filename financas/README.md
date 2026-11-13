# 💰 Minhas Finanças

Sistema pessoal para acompanhar todo o fluxo de dinheiro (contas, cartão e investimentos), com o máximo de automação possível.
Roda 100% no navegador, sem servidor e sem instalação: **abra o `index.html`**.

## O que tem

**Visão geral** (painel principal)
- Resultado do período, receitas, despesas e saldo em conta, com navegação entre meses
- *Saídas por dia*: calendário de calor mostrando quanto saiu em cada dia (clique num dia para ver as transações)
- *Por tags*: rosca de despesas/receitas por categoria (clique numa tag para filtrar)
- Receitas × despesas dos últimos 6 meses, resumo dos investimentos e maiores gastos do mês
- **Minhas contas**: saldo de cada conta e saldo geral; **cartões** com limite disponível e fatura atual; botão *Ajustar saldos* para bater com o app do banco
- **Metas do mês**: limite de gasto por tag com barra de progresso (sugere valores pela média dos últimos 3 meses)
- **Equilíbrio financeiro**: regra 50/30/20 — essenciais até 50% da receita, não essenciais até 30%, investimentos no mínimo 20%
- **Acesso rápido**: lançar despesa, receita, transferência entre contas ou investimento
- **Primeiros passos**: checklist de configuração com progresso
- 👁 **Esconder valores** com um clique (para abrir em público)
- No celular: navegação inferior com botão ＋ para lançamentos rápidos

**Investimentos**
- Carteira consolidada: preço médio, custo, valor atual, resultado (R$ e %) e peso de cada ativo
- Classes: Ações, FIIs, ETFs, BDRs, Renda Fixa, Tesouro Direto, Cripto, Exterior
- Compras, vendas (com lucro realizado) e proventos (dividendos, JCP, rendimentos)
- Alocação por classe, aportes por mês e evolução do patrimônio
- **Cotações automáticas**: B3 via [brapi.dev](https://brapi.dev) (token gratuito em Configurações) e cripto via CoinGecko
- Renda fixa/Tesouro: informe o valor atual de vez em quando (✎)

**Atualização automática (sem clicar em nada)**
- Cotações de ações, FIIs, ETFs, BDRs (brapi.dev) e cripto (CoinGecko) ao abrir o app, a cada 20 min no pregão e ao voltar para ele
- **Tesouro Direto pelo preço de mercado** (PU do dia × quantidade de títulos) via brapi.dev
- **Renda fixa e Tesouro pelo indexador** (% do CDI, prefixado ou IPCA+), com CDI e IPCA 12 meses buscados no Banco Central
- Preços ficam num cache local e não disputam a sincronização com lançamentos feitos em outro aparelho

**Análise da carteira** (não é recomendação: mostra fatos para apoiar a decisão)
- Rebalanceamento por % alvo de cada classe, inclusive só com aportes (sem vender)
- Sinais por ativo: rendimento × CDI no período, concentração, distância das máximas/mínimas de 52 semanas, dividend yield de FIIs × CDI líquido, prejuízos que compensam IR
- Simulador de venda/troca com IR estimado (isenção de R$ 20 mil em ações, R$ 35 mil em cripto, 20% em FIIs…) e carteira resultante

**Estratégias**
- Financiamento: amortizar × investir, reserva de emergência, meta de quitação (quanto amortizar por mês), FGTS a cada 24 meses e comparação automática das estratégias
- Consórcio: chance por sorteio, análise dos lances vencedores, lance recomendado, poder de lance e plano para chegar lá

**Financiamentos e consórcios**
- Financiamento imobiliário **SAC ou Price**: saldo devedor, parcelas pagas/restantes, próxima parcela, juros a pagar, data de quitação, gráfico e tabela de parcelas
- **Simulador completo de amortização**: monte um plano com valores únicos ou recorrentes (todo mês, 13º, FGTS a cada 2 anos…), reduzindo prazo ou parcela, e compare *hoje × com o plano* (quitação, juros, economia, gráficos e tabela ano a ano). O plano pode ser salvo
- **Concretizar amortizações**: "✔ Realizei" registra a amortização e, depois, você informa o que o banco mostrou (saldo devedor, nova parcela, parcelas restantes) — a tabela passa a seguir os números reais
- **Atualizar pelos dados do banco** a qualquer momento (inclui TR e correções) e ver quanto as amortizações já economizaram
- **Consórcios**: parcela com taxa de administração, fundo de reserva, seguro e reajuste anual; parcelas pagas, falta pagar, contemplação e **simulador de lance** (inclusive embutido)
- Parcelas lançadas automaticamente todo mês (valor calculado pela tabela) e **patrimônio líquido** (contas + investimentos + bens − dívidas)

**Ícones**
- Ícone por categoria (escolha no seletor) e por classe de investimento
- Selo com cor e sigla dos principais bancos, detectado pelo nome da conta — ou envie uma imagem própria

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

## Sincronização entre aparelhos (PC ↔ celular)

Por padrão os dados ficam no `localStorage` do navegador de cada aparelho. Para usar os mesmos dados em todos:

1. Crie um token em <https://github.com/settings/tokens/new?scopes=gist&description=Minhas%20Finan%C3%A7as> com **somente a permissão `gist`** (Expiration: *No expiration*, ou renove quando vencer).
2. No app, vá em **Configurações › Sincronização**, cole o token e crie uma **senha de sincronização**.
3. Em cada outro aparelho, faça o mesmo com **o mesmo token e a mesma senha** — o app encontra os dados sozinho.

Como funciona:
- Os dados são comprimidos e **criptografados no próprio aparelho** (PBKDF2 + AES‑256‑GCM) com a sua senha e guardados num **Gist secreto** da sua conta do GitHub. O GitHub só vê texto cifrado.
- Cada alteração é enviada ~3 s depois; ao abrir o app ou voltar para ele, as novidades são baixadas. O indicador ☁ no topo mostra o estado.
- Se os dois aparelhos mudarem ao mesmo tempo sem conexão, vale a versão mais recente e a outra fica guardada em **Configurações › Backup** para restaurar.
- **Se esquecer a senha, os dados da nuvem não podem ser abertos** (os de cada aparelho continuam lá). Token e senha ficam salvos só no aparelho.

## Dados e privacidade

Sem sincronização, nada sai do navegador (exceto os tickers, ao buscar cotações). Com ela, só sai o arquivo criptografado.
Use **Configurações › Backup › Exportar JSON** de vez em quando como cópia de segurança extra.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `index.html` | estrutura da página |
| `styles.css` | visual (claro/escuro automático, responsivo) |
| `app.js` | estado, cálculos, telas, regras, recorrências, carteira e cotações |
| `importers.js` | leitores de OFX, CSV bancário e CSV de operações |
| `sync.js` | sincronização criptografada via GitHub Gist |
