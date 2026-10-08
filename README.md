# Case Técnico de Ciência de Dados: risco de crédito

Solução para o case de concessão de crédito a pessoa física. Entrega um modelo que estima a probabilidade de inadimplência de cada pedido da `base_submissao.parquet` e uma política de crédito que usa essa probabilidade para decidir.

## O que está na entrega

| Arquivo | O que é |
|---|---|
| `submissao_case.csv` | Previsões: `id_cliente` e `probabilidade_inadimplencia`, uma linha para cada um dos 40.000 pedidos |
| `script.py` | Gera o `submissao_case.csv` do início ao fim, a partir das bases brutas |
| `01_EDA_and_Feature_engineering.ipynb` | Exploração das quatro bases, definição do target e construção das features |
| `02_modelagem.ipynb` | Comparação de modelos, tuning, teste fora do tempo, calibração e interpretação |
| `03_politica_credito.ipynb` | Política de crédito baseada em RAROC por faixa de score |
| `target_engineering.py` | Classe `TargetEngineering`: constrói o target |
| `feature_engineering.py` | Classe `FeatureEngineering`: constrói as features |
| `modelagem.py` | Split temporal, modelos, tuning, calibração e métricas |
| `politica_credito.py` | Cálculo do RAROC, cenários de risco e regras da política |
| `dataviz.py` | Gráficos da exploração |
| `requirements.txt` | Dependências, com as versões usadas |

Arquivos gerados ao executar os notebooks, que não fazem parte da entrega:

| Arquivo | Gerado por | O que é |
|---|---|---|
| `data/processed/base_modelagem.parquet` | notebook `01` | Contratos aprovados do histórico, com features e targets |
| `data/processed/base_score.parquet` | notebook `01` | Pedidos da submissão, com as features |
| `outputs/modelo_final.json` | notebook `02` | Configuração e métricas do modelo final |
| `outputs/politica_submissao.csv` | notebook `03` | Decisão da política para cada pedido |

A lógica fica nos arquivos `.py`. Os notebooks instanciam as classes e trazem a explicação da abordagem, com a leitura de cada resultado.

## Como reproduzir

**Versão da linguagem:** Python 3.13.5.

1. Instale as dependências:

   ```
   pip install -r requirements.txt
   ```

2. Coloque as quatro bases do case na pasta `data/`:

   ```
   data/base_cadastral.parquet
   data/base_submissao.parquet
   data/historico_emprestimos.parquet
   data/historico_parcelas.parquet
   data/dicionario_dados.csv
   ```

3. Gere as previsões, a partir da pasta do projeto:

   ```
   python script.py
   ```

   O script leva menos de um minuto e grava o `submissao_case.csv`. Ele não depende dos notebooks nem de bases intermediárias.

Para reproduzir também as análises, execute os notebooks em ordem. O `01` grava as bases intermediárias em `data/processed/`, que o `02` e o `03` leem.

| Notebook | Tempo aproximado |
|---|---|
| `01_EDA_and_Feature_engineering.ipynb` | 3 minutos |
| `02_modelagem.ipynb` | 6 minutos, dos quais 4 são o tuning |
| `03_politica_credito.ipynb` | 30 segundos |

O `script.py` e o notebook `02` geram o mesmo `submissao_case.csv`. A diferença é que o notebook refaz a escolha do modelo e o tuning, e o script usa direto o limiar de atraso e os hiperparâmetros que o tuning escolheu.

## Abordagem

### Definição de inadimplência

O target é o **EVER30MOB12**: o contrato atingiu 30 ou mais dias de atraso em algum momento dos 12 primeiros meses. A justificativa completa está na seção 5 do notebook `01`.

- **Janela de 12 meses.** É o horizonte padrão da probabilidade de default (Basileia), e a curva de safra mostra que é tempo suficiente para a inadimplência aparecer.
- **Limiar de 30 dias, e não 90.** O default de Basileia (90 dias) tem só 395 casos em cerca de 90 mil contratos. Com 30 dias são 1.947. O modelo treinado com o limiar menor ordenou melhor até o próprio default de 90 dias em safras futuras. O EVER90MOB12 continua calculado, como referência.
- **Custo dessa escolha.** A maior parte dos atrasos de 30 dias é paga depois, então o target mistura atraso temporário com default. A escolha foi empírica, pelo número de eventos.
- **Só contratos com a janela completa são rotulados.** A data de observação é a do pedido de cada cliente na submissão, porque o histórico de cada cliente termina ali.
- **Parcelas não pagas não aparecem na base.** A tabela de parcelas registra quase só pagamentos. Para contratos abertos que pararam de pagar, criamos a próxima parcela esperada, sem pagamento. Sem isso, quem nunca mais pagou seria rotulado como bom.

### Features

Cada linha usa só o que era conhecido na data do pedido: cadastro, tipo de contrato, dia e hora da solicitação e o histórico do cliente até aquela data (pedidos anteriores, recusas, atrasos e parcelas em aberto). São 42 features.

Ficaram de fora:

- `sexo`, por equidade;
- oito features de tamanho e prazo do pedido. No histórico elas descrevem um valor aprovado; na submissão, um valor pedido, que é maior. Com elas, o cliente sem histórico recebia a mesma PD dos demais, embora atrase cerca de duas vezes mais.

### Modelo

- **XGBoost**, comparado com uma regressão logística com WoE e com o HistGradientBoosting.
- **Split temporal:** treino até junho de 2023 e teste de julho de 2023 a fevereiro de 2024. O modelo, o limiar e os hiperparâmetros foram escolhidos com cortes dentro do treino; o teste não participou de nenhuma escolha.
- **Treino com limiar de 7 dias de atraso.** Há 10.382 maus com 7 dias, contra 1.848 com 30, e o modelo treinado assim ordena melhor o EVER30. O limiar entrou no tuning como mais um hiperparâmetro.
- **Calibração na safra mais recente**, com um nível por tipo de contrato. A probabilidade entregue é a de EVER30MOB12.

Desempenho no teste fora do tempo (16.446 contratos, 99 maus):

| Medida | Valor |
|---|---|
| AUC no EVER30MOB12 | 0,73 |
| Gini no EVER30MOB12 | 0,46 (intervalo de 95%: 0,38 a 0,55) |
| Gini no EVER90MOB12 | 0,56 (15 casos) |
| Gini só em Cash loans | 0,45 |

A PD média na submissão é de 0,62%.

### Política de crédito

O critério é o **RAROC** (retorno ajustado ao risco sobre o capital) de cada faixa de score, comparado ao retorno mínimo exigido. Os pedidos são divididos em 5 faixas pela PD do modelo (faixa 1 = menor risco).

A base só tem clientes que voltaram a pedir crédito, e a perda observada neles é quase nula. Por isso o nível de perda vem de cenários, que são premissas: default médio de 5% (mercado) e de 10% (estresse). O modelo define a ordenação entre os clientes; o cenário define o nível.

| Grupo | Decisão |
|---|---|
| Faixas 1 a 4 | Aprovar nas condições atuais |
| Faixa 5 | Aprovar com taxa 0,2 ponto percentual ao mês maior e limite de 70% do valor pedido |
| Clientes sem histórico, em qualquer faixa | Limite de 70% do valor pedido |

- Nenhuma recusa: 78,6% dos pedidos são aprovados sem alteração e 21,4% com condições.
- No cenário de mercado, todas as faixas pagam o retorno mínimo. Só a faixa 5 fica abaixo dele, e só no estresse.
- Ajustar a taxa da faixa 5 rende mais do que recusá-la. Recusar aumenta o RAROC médio, mas reduz o lucro econômico.
- O limite de crédito não melhora o retorno. É uma regra de prudência para os grupos em que a estimativa de risco é menos confiável.

Custo de captação, custo operacional, retorno mínimo, LGD e nível de default são premissas e devem ser trocados pelos números do banco. O notebook `03` mostra quanto o resultado depende de cada uma.

## Principais limitações

- **Poucos maus.** O teste tem 99, e os intervalos de confiança dos três modelos se sobrepõem.
- **Viés de seleção.** O treino só tem contratos aprovados, de clientes que voltaram a pedir crédito. A inadimplência observada é baixa e não representa quem não voltou.
- **Nível da PD.** O nível dos Cash loans, que são 91% da submissão, vem de 39 maus do teste. É preciso recalibrar à medida que novas safras completarem 12 meses.
- **Clientes sem histórico** são mais difíceis de ordenar: o modelo tem praticamente só o cadastro deles.
- **A política depende de premissas** que não existem nas bases, e supõe que mudar a taxa ou o limite não muda a PD do cliente.

As limitações e os trabalhos futuros estão detalhados no fim dos notebooks `02` e `03`.
