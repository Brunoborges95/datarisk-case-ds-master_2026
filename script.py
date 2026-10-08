"""Gera ``submissao_case.csv`` a partir das bases brutas da pasta ``data/``.

Uso (a partir da pasta do projeto):

    python script.py

Etapas, as mesmas dos notebooks 01 e 02, sem graficos nem comparacoes:

1. Le as quatro bases.
2. Constroi o target dos contratos aprovados do historico.
3. Constroi as features do historico (treino) e da submissao (score).
4. Separa treino e teste por data e treina o XGBoost.
5. Calibra a PD no teste, com um nivel por tipo de contrato.
6. Pontua a base de submissao e grava o arquivo.

O limiar do target de treino e os hiperparametros foram escolhidos no tuning
do notebook ``02_modelagem.ipynb`` (secao 4.1). Aqui eles entram fixos, para
o script nao repetir a busca, que leva alguns minutos.
"""

from pathlib import Path

import pandas as pd

from feature_engineering import FeatureEngineering
from modelagem import CalibradorPlatt, ModeloCalibrado, ModeloXGBoost, SplitTemporal, avaliar
from target_engineering import TargetEngineering

PASTA_DADOS = Path("data")
ARQUIVO_SAIDA = Path("submissao_case.csv")

# A PD entregue e a de EVER30MOB12: 30 ou mais dias de atraso nos 12 primeiros meses.
DIAS_ATRASO_TARGET = 30
# O modelo aprende com um limiar menor (7 dias), que tem mais maus e ordenou melhor o proprio EVER30.
DIAS_ATRASO_TREINO = 7
MESES_JANELA = 12
INICIO_TESTE = "2023-07-01"

# Colunas que nao entram no modelo.
# - sexo: fica fora por equidade.
# - tamanho e prazo do pedido: no treino descrevem um valor aprovado; na submissao, um valor pedido.
FORA_DO_MODELO = [
    "sexo", "valor_credito", "valor_bem", "valor_parcela", "prazo_estimado_meses",
    "parcela_sobre_credito", "credito_sobre_bem", "credito_sobre_renda", "parcela_sobre_renda",
]

# Melhor tentativa do tuning (notebook 02, secao 4.1).
HIPERPARAMETROS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.041555201306153415,
    "min_child_weight": 4.6125002756572195,
    "reg_lambda": 4.338538352255503,
    "subsample": 0.7564271279729996,
    "colsample_bytree": 0.9725682721151931,
}


def ler_bases() -> dict[str, pd.DataFrame]:
    nomes = ["base_cadastral", "base_submissao", "historico_emprestimos", "historico_parcelas"]
    return {nome: pd.read_parquet(PASTA_DADOS / f"{nome}.parquet") for nome in nomes}


def construir_targets(bases: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str, str]:
    """Uma linha por contrato aprovado, com o target entregue e o target de treino."""
    # O historico de cada cliente termina na data do pedido dele na submissao.
    observacao_por_cliente = bases["base_submissao"].set_index("id_cliente")["data_solicitacao"]
    te = TargetEngineering(bases["historico_emprestimos"], bases["historico_parcelas"], observacao_por_cliente)

    entregue = te.construir_target(DIAS_ATRASO_TARGET, MESES_JANELA)
    treino = te.construir_target(DIAS_ATRASO_TREINO, MESES_JANELA)
    coluna_target, coluna_treino = entregue.columns[-1], treino.columns[-1]

    targets = entregue[["id_contrato", coluna_target]].copy()
    targets[coluna_treino] = treino[coluna_treino].to_numpy()
    return targets, coluna_target, coluna_treino


def main() -> None:
    print("1/6 Lendo as bases...")
    bases = ler_bases()

    print("2/6 Construindo o target...")
    targets, coluna_target, coluna_treino = construir_targets(bases)

    print("3/6 Construindo as features...")
    fe = FeatureEngineering(bases["historico_emprestimos"], bases["historico_parcelas"], bases["base_cadastral"])
    # Treino: contratos aprovados, com as features conhecidas na data da decisao.
    aprovados = bases["historico_emprestimos"].merge(targets, on="id_contrato")
    base_modelagem = fe.construir_base(aprovados, "data_decisao")
    # Contratos sem 12 meses completos nao tem target e ficam fora.
    base_modelagem = base_modelagem[base_modelagem[coluna_target].notna()].reset_index(drop=True)
    # Score: pedidos da submissao, com as features conhecidas na data da solicitacao.
    base_score = fe.construir_base(bases["base_submissao"], "data_solicitacao")

    features = [c for c in base_score.columns if c not in ["id_cliente", "data_referencia", *FORA_DO_MODELO]]
    categoricas = [c for c in features if not pd.api.types.is_numeric_dtype(base_score[c])]
    print(f"    {len(base_modelagem):,} contratos rotulados | {len(base_score):,} pedidos a pontuar | {len(features)} features")

    print("4/6 Treinando o XGBoost...")
    treino, teste = SplitTemporal(INICIO_TESTE).separar(base_modelagem)
    modelo = ModeloXGBoost(categoricas, **HIPERPARAMETROS).fit(treino[features], treino[coluna_treino])
    score_teste = modelo.predict_proba(teste[features])
    metricas = avaliar(teste[coluna_target], score_teste)
    print(f"    treino: {len(treino):,} contratos | teste: {len(teste):,} contratos, {int(teste[coluna_target].sum())} maus")
    print(f"    teste fora do tempo: AUC {metricas['AUC']:.3f} | Gini {metricas['Gini']:.3f} | KS {metricas['KS']:.3f}")

    print("5/6 Calibrando a PD no teste, por tipo de contrato...")
    calibrador = CalibradorPlatt().fit(score_teste, teste[coluna_target], teste["tipo_contrato"])
    modelo_final = ModeloCalibrado(modelo, calibrador, coluna_grupo="tipo_contrato")

    print("6/6 Pontuando a submissao...")
    submissao = pd.DataFrame({
        "id_cliente": base_score["id_cliente"],
        "probabilidade_inadimplencia": modelo_final.predict_proba(base_score[features]),
    })
    assert len(submissao) == len(bases["base_submissao"]) and submissao["id_cliente"].is_unique
    assert submissao["probabilidade_inadimplencia"].between(0, 1).all()
    submissao.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"{ARQUIVO_SAIDA} salvo: {len(submissao):,} linhas | PD media {submissao['probabilidade_inadimplencia'].mean():.2%}")


if __name__ == "__main__":
    main()
