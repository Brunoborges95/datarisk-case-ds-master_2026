"""Separacao temporal, modelos e avaliacao.

Conteudo
--------
- ``SplitTemporal``: treino / teste por data do pedido.
- ``WoEEncoder``: transforma cada feature no WoE da sua faixa (scorecard).
- ``ModeloLogistico``: WoE + regressao logistica (padrao de mercado).
- ``ModeloHistGradient``: gradient boosting por histogramas do scikit-learn.
- ``ModeloXGBoost``: gradient boosting do XGBoost.
- ``TuningXGBoost``: busca bayesiana dos hiperparametros do XGBoost e do
  limiar de atraso do target de treino.
- ``CalibradorPlatt`` e ``ModeloCalibrado``: ajustam o nivel da PD na safra
  mais recente, com um nivel unico ou um por grupo (tipo de contrato).
- ``comparar_calibracao_cv``: compara a calibracao global com a por grupo.
- Funcoes de avaliacao, interpretacao e graficos (AUC, Gini, KS, calibracao,
  faixas, importancia por permutacao, dependencia parcial).

Todos os modelos seguem a mesma interface: ``fit(X, y)`` e
``predict_proba(X)``, que devolve a probabilidade de mau (1D).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

SEED = 42
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


# ======================================================================
# Separacao temporal
# ======================================================================
class SplitTemporal:
    """Separa a base em treino e teste pela data do pedido, nunca aleatoriamente.

    O modelo sera usado em pedidos futuros, entao a avaliacao honesta e
    treinar no passado e medir nas safras mais recentes.
    """

    def __init__(self, inicio_teste: str, coluna_data: str = "data_referencia"):
        self.inicio_teste = pd.Timestamp(inicio_teste)
        self.coluna_data = coluna_data

    def separar(self, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = base[self.coluna_data]
        return base[data < self.inicio_teste], base[data >= self.inicio_teste]

    def resumo(self, partes: dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "periodo": nome,
                "de": parte[self.coluna_data].min().date(),
                "ate": parte[self.coluna_data].max().date(),
                "contratos": len(parte),
                "maus": int(parte[target].sum()),
                "taxa_maus_%": round(parte[target].mean() * 100, 2),
            }
            for nome, parte in partes.items()
        ]).set_index("periodo")


# ======================================================================
# Metricas
# ======================================================================
def ks(y, p) -> float:
    """Maior distancia entre as distribuicoes acumuladas de bons e maus."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def avaliar(y, p) -> dict[str, float]:
    """Metricas padrao de credito: ordenacao (AUC, Gini, KS) e nivel (Brier, PD media)."""
    y = np.asarray(y, dtype=int)
    auc = roc_auc_score(y, p)
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks(y, p),
        "AUC_PR": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "PD_media_prevista": float(np.mean(p)),
        "taxa_maus_real": float(y.mean()),
    }


def ic_gini(y, p, repeticoes: int = 300) -> tuple[float, float]:
    """Intervalo de 95% do Gini por bootstrap: reamostra os contratos e recalcula.

    Com poucos maus o Gini muda bastante de uma amostra para outra; o
    intervalo mostra quanto do numero e ruido.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(SEED)
    ginis = []
    for _ in range(repeticoes):
        amostra = rng.integers(0, len(y), len(y))
        if 0 < y[amostra].sum() < len(amostra):
            ginis.append(2 * roc_auc_score(y[amostra], p[amostra]) - 1)
    inferior, superior = np.percentile(ginis, [2.5, 97.5])
    return float(inferior), float(superior)


def gini_em_cortes_internos(fabricas: dict, treino: pd.DataFrame, features: list[str], targets_treino: list[str],
                            target_avaliacao: str, cortes: list[str]) -> pd.DataFrame:
    """Gini de cada (modelo, target de treino) em cortes temporais DENTRO do treino.

    Para cada corte, treina com os contratos anteriores usando o target de treino
    e mede o Gini em ``target_avaliacao`` nos contratos posteriores, na carteira
    toda e so em Cash loans. O teste nao entra, entao uma escolha feita com esta
    tabela nao contamina a avaliacao final.
    """
    linhas = []
    for corte in cortes:
        passado, futuro = SplitTemporal(corte).separar(treino)
        cash = futuro["tipo_contrato"].eq("Cash loans").to_numpy()
        for nome, criar in fabricas.items():
            for target_treino in targets_treino:
                previsao = criar().fit(passado[features], passado[target_treino]).predict_proba(futuro[features])
                linhas.append({
                    "modelo": nome, "treinado_com": target_treino, "corte": corte,
                    "Gini": avaliar(futuro[target_avaliacao], previsao)["Gini"],
                    "Gini_cash": avaliar(futuro.loc[cash, target_avaliacao], previsao[cash])["Gini"],
                })
    return pd.DataFrame(linhas)


def tabela_faixas(p, y, y_basileia=None, n_faixas: int = 10) -> pd.DataFrame:
    """Contratos agrupados em faixas de score (1 = menor risco). Base da politica de credito."""
    faixa = pd.qcut(pd.Series(p).rank(method="first"), n_faixas, labels=range(1, n_faixas + 1))
    tabela = pd.DataFrame({"faixa": faixa, "pd": p, "mau": np.asarray(y, dtype=float)})
    if y_basileia is not None:
        tabela["basileia"] = np.asarray(y_basileia, dtype=float)
    resumo = tabela.groupby("faixa", observed=True).agg(
        contratos=("pd", "size"), pd_media=("pd", "mean"), taxa_maus=("mau", "mean"),
        **({"taxa_ever90": ("basileia", "mean")} if y_basileia is not None else {}),
    )
    resumo["maus_acumulados_%"] = (
        tabela.groupby("faixa", observed=True)["mau"].sum()[::-1].cumsum()[::-1] / tabela["mau"].sum()
    )
    return resumo


# ======================================================================
# WoE + regressao logistica
# ======================================================================
class WoEEncoder:
    """Troca cada feature pelo Weight of Evidence da faixa em que o valor cai.

    - Numericas: faixas por quantis aprendidos no treino; nulo vira faixa propria.
    - Categoricas: categorias raras viram "Outros"; nulo vira faixa propria.
    - WoE = ln(% de bons na faixa / % de maus na faixa); positivo = menos risco.
    - Features com IV abaixo de ``iv_minimo`` sao descartadas.
    """

    def __init__(self, n_faixas: int = 8, iv_minimo: float = 0.02, participacao_minima: float = 0.01):
        self.n_faixas = n_faixas
        self.iv_minimo = iv_minimo
        self.participacao_minima = participacao_minima

    def _agrupar(self, x: pd.Series, ajustar: bool) -> pd.Series:
        nome = x.name
        if pd.api.types.is_numeric_dtype(x):
            if ajustar:
                cortes = np.unique(np.nanquantile(x.astype(float), np.linspace(0, 1, self.n_faixas + 1)))
                self.cortes_[nome] = np.r_[-np.inf, cortes[1:-1], np.inf]
            grupos = pd.cut(x, self.cortes_[nome], include_lowest=True).astype(str)
        else:
            if ajustar:
                frequencia = x.value_counts(normalize=True)
                self.categorias_[nome] = set(frequencia[frequencia >= self.participacao_minima].index)
            grupos = x.where(x.isin(self.categorias_[nome]) | x.isna(), "Outros").astype(str)
        return grupos.where(x.notna(), "Nulo")

    def fit(self, X: pd.DataFrame, y) -> "WoEEncoder":
        y = pd.Series(np.asarray(y, dtype=int), index=X.index)
        self.cortes_, self.categorias_, self.woe_, iv = {}, {}, {}, {}
        for coluna in X.columns:
            tabela = pd.crosstab(self._agrupar(X[coluna], ajustar=True), y).reindex(columns=[0, 1], fill_value=0)
            bons = (tabela[0] + 0.5) / tabela[0].sum()
            maus = (tabela[1] + 0.5) / tabela[1].sum()
            self.woe_[coluna] = np.log(bons / maus)
            iv[coluna] = float(((bons - maus) * self.woe_[coluna]).sum())
        self.iv_ = pd.Series(iv).sort_values(ascending=False)
        self.features_ = list(self.iv_[self.iv_ >= self.iv_minimo].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: self._agrupar(X[c], ajustar=False).map(self.woe_[c]).fillna(0.0) for c in self.features_},
            index=X.index,
        )


class ModeloLogistico:
    """Scorecard: WoE + regressao logistica. Sem peso de classe, para a PD sair calibrada."""

    nome = "Regressão logística (WoE)"

    def __init__(self, n_faixas: int = 8, iv_minimo: float = 0.02, C: float = 1.0):
        self.encoder = WoEEncoder(n_faixas=n_faixas, iv_minimo=iv_minimo)
        self.modelo = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X, y) -> "ModeloLogistico":
        self.modelo.fit(self.encoder.fit(X, y).transform(X), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.modelo.predict_proba(self.encoder.transform(X))[:, 1]

    def coeficientes(self) -> pd.DataFrame:
        """Peso de cada feature. Coeficiente negativo e o esperado: WoE alto = menos risco."""
        return pd.DataFrame({
            "IV": self.encoder.iv_[self.encoder.features_],
            "coeficiente": self.modelo.coef_[0],
        }).sort_values("IV", ascending=False)


# ======================================================================
# Gradient boosting: HistGradientBoosting e XGBoost
# ======================================================================
class ModeloArvores:
    """Base dos dois modelos de gradient boosting.

    Os dois aceitam nulos e categoricas sem pre-processamento, desde que as
    categoricas cheguem como ``category`` do pandas. As categorias sao fixadas
    no treino; uma categoria nova na hora de prever vira nulo.
    """

    nome = "Gradient boosting"

    def __init__(self, categoricas: list[str], **params):
        self.categoricas = categoricas
        self.params = params

    def _criar_modelo(self):
        raise NotImplementedError

    def _preparar(self, X: pd.DataFrame, ajustar: bool = False) -> pd.DataFrame:
        X = X.copy()
        if ajustar:
            self.categorias_ = {c: sorted(X[c].dropna().unique()) for c in self.categoricas}
        for coluna in self.categoricas:
            X[coluna] = pd.Categorical(X[coluna], categories=self.categorias_[coluna])
        return X

    def fit(self, X, y) -> "ModeloArvores":
        self.modelo = self._criar_modelo().fit(self._preparar(X, ajustar=True), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.modelo.predict_proba(self._preparar(X))[:, 1]


class ModeloHistGradient(ModeloArvores):
    """Gradient boosting do scikit-learn: agrupa cada feature em faixas (histograma) antes de crescer as arvores."""

    nome = "HistGradientBoosting"

    def _criar_modelo(self):
        return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED, **self.params)


class ModeloXGBoost(ModeloArvores):
    """Gradient boosting do XGBoost, com regularizacao L2 nas folhas e amostragem de linhas e colunas."""

    nome = "XGBoost"

    def _criar_modelo(self):
        return XGBClassifier(
            enable_categorical=True, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1, **self.params
        )


# ======================================================================
# Tuning do XGBoost
# ======================================================================
class TuningXGBoost:
    """Busca bayesiana (Optuna, amostrador TPE) dos hiperparametros do XGBoost.

    O limiar de dias de atraso do target de TREINO entra na busca como mais um
    hiperparametro. Para isso, o treino precisa ter uma coluna
    ``target_ever{dias}mob12`` para cada limiar entre ``dias_min`` e ``dias_max``.

    Cada tentativa e avaliada com ``gini_em_cortes_internos``: treina com os
    contratos anteriores a cada corte e mede o Gini do target de avaliacao nos
    posteriores, sempre dentro do treino. O teste nao participa.

    O numero de arvores fica fixo: a taxa de aprendizado, que esta na busca,
    ja controla o quanto o modelo aprende.
    """

    def __init__(self, treino: pd.DataFrame, features: list[str], categoricas: list[str], target_avaliacao: str,
                 cortes: list[str], n_arvores: int = 400, dias_min: int = 5, dias_max: int = 30):
        self.treino = treino
        self.features = features
        self.categoricas = categoricas
        self.target_avaliacao = target_avaliacao
        self.cortes = cortes
        self.n_arvores = n_arvores
        self.dias_min = dias_min
        self.dias_max = dias_max

    @staticmethod
    def coluna_target(dias: int) -> str:
        """Nome da coluna do target com limiar de ``dias`` de atraso em 12 meses."""
        return f"target_ever{dias}mob12"

    def _objetivo(self, trial) -> float:
        """Uma tentativa: sorteia limiar e hiperparametros e devolve o Gini medio dos cortes internos."""
        dias = trial.suggest_int("dias_atraso", self.dias_min, self.dias_max, log=True)
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        fabrica = {"XGBoost": lambda: ModeloXGBoost(self.categoricas, n_estimators=self.n_arvores, **params)}
        resultado = gini_em_cortes_internos(
            fabrica, self.treino, self.features, [self.coluna_target(dias)], self.target_avaliacao, self.cortes
        )
        trial.set_user_attr("Gini_cash", float(resultado["Gini_cash"].mean()))
        return float(resultado["Gini"].mean())

    def executar(self, n_tentativas: int, ponto_de_partida: dict) -> "TuningXGBoost":
        """Roda a busca. ``ponto_de_partida`` (limiar + hiperparametros) e avaliado como a primeira tentativa."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.estudo_ = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        self.estudo_.enqueue_trial(ponto_de_partida)
        self.estudo_.optimize(self._objetivo, n_trials=n_tentativas)

        melhor = dict(self.estudo_.best_params)
        self.melhor_dias_ = melhor.pop("dias_atraso")
        self.melhores_parametros_ = {"n_estimators": self.n_arvores, **melhor}
        return self

    def tentativas(self) -> pd.DataFrame:
        """Uma linha por tentativa, da melhor para a pior. A tentativa 0 e o ponto de partida."""
        tabela = self.estudo_.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        tabela.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in tabela.columns]
        tabela = tabela.rename(columns={"number": "tentativa", "value": "Gini"}).set_index("tentativa")
        return tabela.sort_values("Gini", ascending=False)


# ======================================================================
# Interpretacao (serve para qualquer modelo)
# ======================================================================
def _log_odds(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def importancia_permutacao(modelo, X: pd.DataFrame) -> pd.Series:
    """Quanto a previsao muda, em log-odds, quando os valores de uma feature sao embaralhados.

    Embaralhar quebra a relacao da feature com o contrato. Se a previsao quase
    nao muda, o modelo nao depende dela. Nao usa o target, entao nao sofre com
    o pequeno numero de maus.
    """
    rng = np.random.default_rng(SEED)
    original = _log_odds(modelo.predict_proba(X))
    mudanca = {}
    for coluna in X.columns:
        embaralhado = X.copy()
        embaralhado[coluna] = rng.permutation(X[coluna].to_numpy())
        mudanca[coluna] = np.abs(_log_odds(modelo.predict_proba(embaralhado)) - original).mean()
    return pd.Series(mudanca).sort_values(ascending=False)


def dependencia_parcial(modelo, X: pd.DataFrame, feature: str, n_pontos: int = 20) -> pd.Series:
    """PD media prevista quando TODOS os contratos recebem o mesmo valor da feature.

    Repetindo para varios valores (do percentil 1 ao 99), a curva mostra como o
    modelo responde a feature com todo o resto mantido igual.
    """
    valores = np.unique(np.nanquantile(X[feature].astype(float), np.linspace(0.01, 0.99, n_pontos)))
    curva = {}
    for valor in valores:
        alterado = X.copy()
        alterado[feature] = valor
        curva[valor] = modelo.predict_proba(alterado).mean()
    return pd.Series(curva, name=feature)


# ======================================================================
# Calibracao
# ======================================================================
class CalibradorPlatt:
    """Reajusta o nivel da PD com uma logistica sobre o log-odds do score.

    Sem ``grupo``, ha uma inclinacao e um nivel (intercepto) unicos: a
    ordenacao nao muda (AUC, KS), so o nivel das probabilidades, que vai para
    a taxa de maus do periodo de calibracao, normalmente o mais recente.

    Com ``grupo`` (por exemplo, o tipo de contrato), a inclinacao continua
    unica, mas cada grupo ganha o seu proprio nivel. A ordem dentro de cada
    grupo nao muda; entre grupos, pode mudar. O primeiro grupo em ordem
    alfabetica e a referencia, e um grupo que nao apareceu no ajuste recebe
    o nivel da referencia.
    """

    def fit(self, p, y, grupo=None) -> "CalibradorPlatt":
        self.grupos_ = None if grupo is None else sorted(pd.unique(pd.Series(grupo).dropna()))
        self.modelo = LogisticRegression(C=1e6).fit(self._matriz(p, grupo), np.asarray(y, dtype=int))
        return self

    def transform(self, p, grupo=None) -> np.ndarray:
        return self.modelo.predict_proba(self._matriz(p, grupo))[:, 1]

    def _matriz(self, p, grupo) -> np.ndarray:
        """Log-odds do score e, se houver grupos, uma coluna 0/1 para cada grupo alem da referencia."""
        colunas = [self._logit(p)]
        if self.grupos_ is not None:
            if grupo is None:
                raise ValueError("Este calibrador foi ajustado com grupos: informe `grupo` tambem ao prever.")
            grupo = np.asarray(grupo)
            colunas += [(grupo == g).astype(float).reshape(-1, 1) for g in self.grupos_[1:]]
        return np.hstack(colunas)

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)


class ModeloCalibrado:
    """Modelo + calibrador juntos: ``predict_proba`` ja devolve a PD calibrada.

    ``coluna_grupo`` e a coluna de X que define o grupo do calibrador, quando ele foi ajustado por grupo.
    """

    def __init__(self, modelo, calibrador: CalibradorPlatt, coluna_grupo: str | None = None):
        self.modelo = modelo
        self.calibrador = calibrador
        self.coluna_grupo = coluna_grupo
        self.nome = f"{modelo.nome} calibrado"

    def predict_proba(self, X) -> np.ndarray:
        grupo = None if self.coluna_grupo is None else X[self.coluna_grupo]
        return self.calibrador.transform(self.modelo.predict_proba(X), grupo)


def comparar_calibracao_cv(p, y, grupo, partes: int = 5, repeticoes: int = 20) -> pd.DataFrame:
    """Calibracao global x por grupo, em validacao cruzada dentro do periodo de calibracao.

    Um calibrador com mais parametros sempre parece melhor nos dados em que foi
    ajustado. Aqui, cada contrato e previsto por calibradores que nao o viram.
    Devolve o log-loss das duas versoes em cada repeticao; menor e melhor.
    """
    p, y, grupo = np.asarray(p, dtype=float), np.asarray(y, dtype=int), np.asarray(grupo)
    estrato = np.array([f"{mau}_{g}" for mau, g in zip(y, grupo)])
    divisor = RepeatedStratifiedKFold(n_splits=partes, n_repeats=repeticoes, random_state=SEED)
    previsto = {"global": np.zeros((repeticoes, len(y))), "por grupo": np.zeros((repeticoes, len(y)))}
    for k, (ajuste, fora) in enumerate(divisor.split(p, estrato)):
        repeticao = k // partes
        previsto["global"][repeticao, fora] = CalibradorPlatt().fit(p[ajuste], y[ajuste]).transform(p[fora])
        previsto["por grupo"][repeticao, fora] = (
            CalibradorPlatt().fit(p[ajuste], y[ajuste], grupo[ajuste]).transform(p[fora], grupo[fora])
        )
    return pd.DataFrame({
        nome: [log_loss(y, np.clip(linha, 1e-6, 1 - 1e-6)) for linha in matriz] for nome, matriz in previsto.items()
    }).rename_axis("repeticao")


# ======================================================================
# Graficos
# ======================================================================
def plot_roc_ks(y, previsoes: dict[str, np.ndarray], titulo: str = "") -> None:
    """Curva ROC e curva KS (distribuicoes acumuladas de bons e maus) para cada modelo."""
    y = np.asarray(y, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (nome, p), cor in zip(previsoes.items(), CORES):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=cor, linewidth=2, label=f"{nome} (AUC {roc_auc_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#898781", linewidth=1)
    axes[0].set(xlabel="Taxa de falsos positivos (bons recusados)", ylabel="Taxa de verdadeiros positivos (maus recusados)",
                title=f"Curva ROC {titulo}")
    axes[0].legend(frameon=False)

    nome, p = next(iter(previsoes.items()))
    ordem = np.argsort(p)
    proporcao = np.arange(1, len(p) + 1) / len(p)
    acum_maus = np.cumsum(y[ordem]) / y.sum()
    acum_bons = np.cumsum(1 - y[ordem]) / (1 - y).sum()
    i = np.argmax(acum_bons - acum_maus)
    axes[1].plot(proporcao, acum_bons, color=CORES[0], linewidth=2, label="Bons (acumulado)")
    axes[1].plot(proporcao, acum_maus, color=CORES[1], linewidth=2, label="Maus (acumulado)")
    axes[1].vlines(proporcao[i], acum_maus[i], acum_bons[i], color="#52514e", linestyle="--")
    axes[1].annotate(f"KS = {acum_bons[i] - acum_maus[i]:.3f}", (proporcao[i], (acum_bons[i] + acum_maus[i]) / 2),
                     xytext=(8, 0), textcoords="offset points", color="#52514e")
    axes[1].set(xlabel="Contratos ordenados do menor para o maior risco", ylabel="Proporção acumulada",
                title=f"Curva KS — {nome} {titulo}")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_calibracao(y, previsoes: dict[str, np.ndarray], titulo: str = "", n_faixas: int = 10) -> None:
    """PD prevista media vs taxa de maus observada, por decil de score."""
    y = np.asarray(y, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    maximo = 0
    for (nome, p), cor in zip(previsoes.items(), CORES):
        real, previsto = calibration_curve(y, p, n_bins=n_faixas, strategy="quantile")
        ax.plot(previsto * 100, real * 100, "o-", color=cor, linewidth=2, markersize=6, label=nome)
        maximo = max(maximo, real.max(), previsto.max())
    ax.plot([0, maximo * 100], [0, maximo * 100], "--", color="#898781", linewidth=1, label="Calibração perfeita")
    ax.set(xlabel="PD prevista média (%)", ylabel="Taxa de maus observada (%)", title=f"Calibração por decil {titulo}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
