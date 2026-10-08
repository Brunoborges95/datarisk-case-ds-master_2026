"""Politica de credito baseada em RAROC.

RAROC (retorno ajustado ao risco sobre o capital) do primeiro ano de um contrato:

    RAROC = (juros - custo de captacao - custo operacional - perda esperada) / capital

    perda esperada = PD x LGD x EAD
    capital        = 8% do saldo devedor medio no ano

Conteudo
--------
- ``taxa_implicita`` / ``prazo_implicito``: Tabela Price ao contrario.
- ``HistoricoCredito``: o que da para ESTIMAR do historico (taxa de juros
  cobrada, momento do default, saldo exposto e perda observada).
- ``Premissas``: o que NAO esta nos dados (captacao, custo operacional,
  capital, retorno minimo).
- ``CenarioRisco``: nivel de default e LGD assumidos; a ordenacao dos clientes
  vem sempre do modelo.
- ``CalculadoraRAROC``: economia de cada pedido e agregacao por faixa.
- ``Politica``: regras por faixa de score (recusar, mudar taxa, prazo, limite).

Hipotese importante: mudar taxa, prazo ou limite NAO muda a PD do cliente.
O modelo associa prazo curto a mais risco por causa do produto, nao por
causalidade, entao recalcular a PD com o prazo alterado seria enganoso.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from target_engineering import TargetEngineering


# ======================================================================
# Tabela Price
# ======================================================================
def taxa_implicita(credito, parcela, prazo, iteracoes: int = 60) -> np.ndarray:
    """Taxa mensal que resolve credito = parcela x (1 - (1+i)^-prazo) / i, por bissecao."""
    credito, parcela, prazo = (np.asarray(x, dtype=float) for x in (credito, parcela, prazo))
    baixo, alto = np.full(credito.shape, 1e-6), np.full(credito.shape, 1.0)
    for _ in range(iteracoes):
        meio = (baixo + alto) / 2
        valor_presente = parcela * (1 - (1 + meio) ** -prazo) / meio
        subir = valor_presente > credito          # parcelas valem mais que o credito: a taxa e maior
        baixo, alto = np.where(subir, meio, baixo), np.where(subir, alto, meio)
    return np.where(parcela * prazo > credito, (baixo + alto) / 2, np.nan)


def prazo_implicito(credito, parcela, taxa: float, minimo: int = 6, maximo: int = 60) -> np.ndarray:
    """Numero de parcelas que fecha a Tabela Price com a taxa informada.

    Quando a parcela nao cobre nem os juros do primeiro mes nao ha solucao;
    nesses casos devolve o prazo maximo.
    """
    juros_sobre_parcela = taxa * np.asarray(credito, dtype=float) / np.asarray(parcela, dtype=float)
    prazo = -np.log(1 - np.clip(juros_sobre_parcela, None, 0.999)) / np.log(1 + taxa)
    prazo = np.where(juros_sobre_parcela >= 1, maximo, prazo)
    return np.clip(np.round(prazo), minimo, maximo)


def soma_saldos(credito, taxa, prazo, meses) -> np.ndarray:
    """Soma dos saldos devedores no inicio de cada um dos primeiros ``meses`` meses."""
    fator = (1 + taxa) ** prazo
    return credito * (meses * fator - ((1 + taxa) ** meses - 1) / taxa) / (fator - 1)


def saldo_devedor(credito, taxa, prazo, parcelas_pagas) -> np.ndarray:
    """Saldo devedor depois de ``parcelas_pagas`` parcelas."""
    fator = (1 + taxa) ** prazo
    return credito * (fator - (1 + taxa) ** parcelas_pagas) / (fator - 1)


def escalar_pd(pd_modelo, taxa_alvo: float, iteracoes: int = 60) -> np.ndarray:
    """Reescala a PD para que a media seja ``taxa_alvo``, preservando a ordenacao.

    Multiplica as chances (odds) de todos os clientes pela mesma constante: quem
    era 3x mais arriscado continua ~3x mais arriscado, so o nivel muda.
    """
    pd_modelo = np.clip(np.asarray(pd_modelo, dtype=float), 1e-9, 1 - 1e-9)
    log_odds = np.log(pd_modelo / (1 - pd_modelo))
    baixo, alto = -20.0, 20.0
    for _ in range(iteracoes):
        meio = (baixo + alto) / 2
        if (1 / (1 + np.exp(-(log_odds + meio)))).mean() < taxa_alvo:
            baixo = meio
        else:
            alto = meio
    return 1 / (1 + np.exp(-(log_odds + (baixo + alto) / 2)))


# ======================================================================
# O que o historico permite estimar
# ======================================================================
class HistoricoCredito:
    """Estimativas tiradas dos contratos parcelados aprovados no historico."""

    def __init__(self, hist_emprestimos: pd.DataFrame, hist_parcelas: pd.DataFrame,
                 observacao_por_cliente: pd.Series | None = None):
        # Mesma data de observacao por cliente usada na construcao do target.
        self.alvo = TargetEngineering(hist_emprestimos, hist_parcelas, observacao_por_cliente)
        contratos = hist_emprestimos[
            hist_emprestimos["status_contrato"].eq("Approved")
            & hist_emprestimos["qtd_parcelas_planejadas"].gt(0)
            & hist_emprestimos["valor_parcela"].gt(0)
            & hist_emprestimos["valor_credito"].gt(0)
        ].copy()
        for coluna in ["data_decisao", "data_encerramento"]:
            contratos[coluna] = pd.to_datetime(contratos[coluna], errors="coerce")
        contratos["taxa_mensal"] = taxa_implicita(
            contratos["valor_credito"], contratos["valor_parcela"], contratos["qtd_parcelas_planejadas"]
        )
        self.contratos = contratos.set_index("id_contrato")

    def taxas_por_ano(self) -> pd.DataFrame:
        """Mediana da taxa mensal implicita por produto e ano da decisao (em %)."""
        ano = self.contratos["data_decisao"].dt.year.rename("ano")
        return self.contratos.groupby([ano, "tipo_contrato"])["taxa_mensal"].median().unstack() * 100

    def taxa_recente(self, tipo_contrato: str = "Cash loans", desde: str = "2024-01-01") -> float:
        recentes = self.contratos[
            self.contratos["tipo_contrato"].eq(tipo_contrato) & (self.contratos["data_decisao"] >= desde)
        ]
        return float(recentes["taxa_mensal"].median())

    def eventos(self, dias_atraso: int = 90, meses: int = 12) -> pd.DataFrame:
        """Um contrato por linha entre os que atingiram ``dias_atraso`` na janela.

        - parcelas_pagas_antes: quantas parcelas estavam quitadas no dia do evento
        - saldo_sobre_credito: saldo devedor nesse dia, como fracao do credito (EAD)
        - perda_sobre_saldo: fracao do saldo que nunca foi paga (LGD observada).
          Contrato encerrado foi quitado (verificado na EDA), entao perde zero;
          contrato aberto perde as parcelas do cronograma que nao foram pagas.
        """
        parcelas = self.alvo.parcelas.merge(
            self.alvo.contratos[["id_contrato", "data_decisao", "data_observacao"]], on="id_contrato"
        )
        parcelas = parcelas[parcelas["id_contrato"].isin(self.contratos.index)]
        fim_janela = parcelas["data_decisao"] + pd.DateOffset(months=meses)
        limite = parcelas["data_vencimento"] + pd.Timedelta(days=dias_atraso)
        atingiu = (
            (parcelas["data_pagamento"].isna() | (parcelas["data_pagamento"] >= limite))
            & (limite <= fim_janela) & (fim_janela <= parcelas["data_observacao"])
        )
        data_evento = limite[atingiu].groupby(parcelas.loc[atingiu, "id_contrato"]).min().rename("data_evento")

        d = parcelas.join(data_evento, on="id_contrato", how="inner")
        d["paga_antes"] = d["data_pagamento"] < d["data_evento"]
        d["paga_depois"] = d["data_pagamento"] >= d["data_evento"]
        eventos = d.groupby("id_contrato")[["paga_antes", "paga_depois"]].sum().join(self.contratos)
        eventos = eventos.rename(columns={"paga_antes": "parcelas_pagas_antes", "paga_depois": "parcelas_pagas_depois"})

        restantes = (eventos["qtd_parcelas_planejadas"] - eventos["parcelas_pagas_antes"]).clip(lower=0)
        taxa = eventos["taxa_mensal"].fillna(0).clip(lower=1e-6)
        saldo = eventos["valor_parcela"] * (1 - (1 + taxa) ** -restantes) / taxa
        eventos["saldo_sobre_credito"] = saldo / eventos["valor_credito"]
        eventos["encerrado"] = eventos["data_encerramento"].notna()
        nao_pagas = (restantes - eventos["parcelas_pagas_depois"]).clip(lower=0).where(~eventos["encerrado"], 0)
        eventos["perda_sobre_saldo"] = (nao_pagas / restantes.where(restantes > 0)).fillna(0)
        eventos["saldo"], eventos["perda"] = saldo, saldo * eventos["perda_sobre_saldo"]
        return eventos

    def resumo_eventos(self, dias_atraso: int = 90, meses: int = 12) -> pd.DataFrame:
        eventos = self.eventos(dias_atraso, meses)
        resumo = eventos.groupby("tipo_contrato").agg(
            contratos=("saldo", "size"),
            parcelas_pagas_antes_mediana=("parcelas_pagas_antes", "median"),
            saldo_sobre_credito_mediana=("saldo_sobre_credito", "median"),
            encerrados_pct=("encerrado", lambda x: x.mean() * 100),
            contratos_com_perda=("perda", lambda x: int((x > 0).sum())),
        )
        resumo["perda_sobre_saldo_pct"] = eventos.groupby("tipo_contrato")["perda"].sum() / eventos.groupby("tipo_contrato")["saldo"].sum() * 100
        return resumo

    def razao_default_sobre_atraso(self, meses: int = 12) -> pd.Series:
        """Dos contratos que atingem 30 dias de atraso na janela, quantos chegam a 90 (por produto)."""
        ever30 = self.alvo.construir_target(30, meses).set_index("id_contrato").iloc[:, -1]
        ever90 = self.alvo.construir_target(90, meses).set_index("id_contrato").iloc[:, -1]
        tabela = pd.DataFrame({"ever30": ever30, "ever90": ever90}).dropna().join(
            self.alvo.contratos.set_index("id_contrato")["tipo_contrato"]
        )
        somas = tabela.groupby("tipo_contrato")[["ever30", "ever90"]].sum()
        return (somas["ever90"] / somas["ever30"]).rename("ever90_sobre_ever30")


# ======================================================================
# Carteira a decidir
# ======================================================================
def montar_carteira(base_score: pd.DataFrame, submissao: pd.DataFrame, taxa_atual: float, n_faixas: int = 5) -> pd.DataFrame:
    """Junta os pedidos com a PD do modelo e acrescenta prazo implicito e faixa de score.

    - ``pd_modelo``: probabilidade entregue em ``submissao_case.csv`` (30+ dias em 12 meses).
    - ``prazo``: a submissao nao traz prazo; ele e deduzido de credito e parcela
      supondo que todos pagam a taxa atual.
    - ``faixa``: grupos de mesmo tamanho da PD, quintis por padrao (1 = menor risco). Faixas relativas nao mudam quando
      o modelo e recalibrado.
    """
    carteira = base_score[["id_cliente", "tipo_contrato", "valor_credito", "valor_parcela", "hist_sem_historico"]].merge(
        submissao.rename(columns={"probabilidade_inadimplencia": "pd_modelo"}), on="id_cliente", validate="one_to_one"
    )
    parcela_tipica = (carteira["valor_parcela"] / carteira["valor_credito"]).median()
    carteira["valor_parcela"] = carteira["valor_parcela"].fillna(carteira["valor_credito"] * parcela_tipica)
    carteira["prazo"] = prazo_implicito(carteira["valor_credito"], carteira["valor_parcela"], taxa_atual)
    carteira["faixa"] = pd.qcut(carteira["pd_modelo"].rank(method="first"), n_faixas, labels=range(1, n_faixas + 1)).astype(int)
    return carteira


# ======================================================================
# Premissas e cenarios
# ======================================================================
@dataclass
class Premissas:
    """Parametros economicos. So a taxa de juros e o momento do default vem dos dados."""

    taxa_juros_mensal: float                    # taxa cobrada hoje (estimada do historico)
    parcelas_pagas_antes_do_default: int        # quando o cliente para de pagar (estimado do historico)
    custo_captacao_mensal: float = 0.010        # ~12,7% a.a.
    custo_operacional_anual: float = 0.04       # % do saldo medio: originacao, atendimento, cobranca
    capital: float = 0.08                       # capital alocado sobre o saldo (Basileia)
    retorno_minimo: float = 0.20                # RAROC minimo exigido, ao ano, antes de impostos


@dataclass
class CenarioRisco:
    """Nivel de perda assumido. A ordenacao entre clientes vem sempre do modelo."""

    nome: str
    default_medio: float                        # taxa de default em 12 meses da carteira inteira
    lgd: dict[str, float] = field(default_factory=dict)   # por tipo de contrato

    def aplicar(self, carteira: pd.DataFrame) -> pd.DataFrame:
        """Acrescenta ``pd_default`` e ``lgd``. Deve receber a carteira INTEIRA (a escala usa a media)."""
        return carteira.assign(
            pd_default=escalar_pd(carteira["pd_modelo"], self.default_medio),
            lgd=carteira["tipo_contrato"].map(self.lgd).astype(float),
        )


# ======================================================================
# Politica
# ======================================================================
@dataclass
class Politica:
    """Regras por faixa de score (1 = menor risco). O que nao e citado fica como hoje."""

    nome: str
    recusar_faixas: tuple[int, ...] = ()
    ajuste_taxa: dict[int, float] = field(default_factory=dict)      # variacao da taxa mensal (ex.: -0.003)
    prazo_maximo: dict[int, int] = field(default_factory=dict)       # em meses
    fator_limite: dict[int, float] = field(default_factory=dict)     # 0.7 = aprova 70% do valor pedido
    fator_limite_sem_historico: float = 1.0

    def aplicar(self, carteira: pd.DataFrame, taxa_atual: float) -> pd.DataFrame:
        """Devolve a carteira com ``aprovado``, ``taxa``, ``prazo`` e ``fator_credito`` de cada pedido."""
        faixa = carteira["faixa"].astype(int)
        fator = faixa.map(self.fator_limite).fillna(1.0)
        fator = fator.where(carteira["hist_sem_historico"].eq(0), np.minimum(fator, self.fator_limite_sem_historico))
        return carteira.assign(
            aprovado=~faixa.isin(self.recusar_faixas),
            taxa=taxa_atual + faixa.map(self.ajuste_taxa).fillna(0.0),
            prazo=np.minimum(carteira["prazo"], faixa.map(self.prazo_maximo).fillna(np.inf)),
            fator_credito=fator,
        )


# ======================================================================
# RAROC
# ======================================================================
class CalculadoraRAROC:
    """Calcula a economia do primeiro ano de cada pedido e agrega.

    A carteira precisa ter: ``valor_credito``, ``prazo``, ``pd_default`` e ``lgd``
    (as duas ultimas vem de ``CenarioRisco.aplicar``). Colunas opcionais
    ``taxa`` e ``fator_credito`` (vindas de ``Politica.aplicar``) substituem a
    taxa atual e o valor pedido.
    """

    COMPONENTES = ["receita_juros", "custo_captacao", "custo_operacional", "perda_esperada", "lucro", "capital"]

    def __init__(self, premissas: Premissas):
        self.premissas = premissas

    def calcular(self, carteira: pd.DataFrame, taxa=None, prazo=None, **premissas_alteradas) -> pd.DataFrame:
        """Uma linha por pedido com os componentes do RAROC (em unidades monetarias)."""
        p = replace(self.premissas, **premissas_alteradas)
        if taxa is None:
            taxa = carteira["taxa"].to_numpy() if "taxa" in carteira else p.taxa_juros_mensal
        if prazo is None:
            prazo = carteira["prazo"].to_numpy()
        prazo = np.asarray(prazo, dtype=float)
        fator = carteira["fator_credito"].to_numpy() if "fator_credito" in carteira else 1.0
        credito = carteira["valor_credito"].to_numpy() * fator
        pd_default, lgd = carteira["pd_default"].to_numpy(), carteira["lgd"].to_numpy()

        meses = np.minimum(12, prazo)                                   # horizonte: primeiro ano
        pagas = np.minimum(p.parcelas_pagas_antes_do_default, meses - 1)  # quando o inadimplente para de pagar
        saldos_ano = soma_saldos(credito, taxa, prazo, meses)
        saldos_ate_default = soma_saldos(credito, taxa, prazo, pagas)

        r = pd.DataFrame(index=carteira.index)
        r["credito"] = credito
        r["saldo_medio"] = saldos_ano / 12
        # quem entra em default deixa de pagar juros a partir dali
        r["receita_juros"] = taxa * (saldos_ate_default + (1 - pd_default) * (saldos_ano - saldos_ate_default))
        r["custo_captacao"] = p.custo_captacao_mensal * saldos_ano
        r["custo_operacional"] = p.custo_operacional_anual / 12 * saldos_ano
        r["ead"] = saldo_devedor(credito, taxa, prazo, pagas)
        r["perda_esperada"] = pd_default * lgd * r["ead"]
        r["lucro"] = r["receita_juros"] - r["custo_captacao"] - r["custo_operacional"] - r["perda_esperada"]
        r["capital"] = p.capital * r["saldo_medio"]
        r["defaults_esperados"] = pd_default
        return r

    def agregar(self, resultado: pd.DataFrame, por=None) -> pd.DataFrame:
        """Soma os componentes (por grupo, se informado) e calcula RAROC e taxas sobre o saldo medio."""
        grupos = resultado.groupby(por, observed=True) if por is not None else resultado.groupby(lambda _: "carteira")
        t = grupos[["credito", "saldo_medio", *self.COMPONENTES]].sum()
        t.insert(0, "pedidos", grupos.size())
        t["default_medio_%"] = grupos["defaults_esperados"].mean() * 100
        t["perda_sobre_saldo_%"] = t["perda_esperada"] / t["saldo_medio"] * 100
        t["raroc_%"] = t["lucro"] / t["capital"] * 100
        t["lucro_economico"] = t["lucro"] - self.premissas.retorno_minimo * t["capital"]
        return t

    def raroc(self, carteira: pd.DataFrame, **kwargs) -> float:
        """RAROC agregado da carteira (ou de um pedaco dela)."""
        r = self.calcular(carteira, **kwargs)
        return float(r["lucro"].sum() / r["capital"].sum())

    def taxa_minima(self, carteira: pd.DataFrame, retorno=None, **kwargs) -> float:
        """Menor taxa mensal que leva o RAROC do grupo ao retorno minimo (bissecao)."""
        retorno = self.premissas.retorno_minimo if retorno is None else retorno
        baixo, alto = 1e-4, 0.20
        for _ in range(50):
            meio = (baixo + alto) / 2
            if self.raroc(carteira, taxa=meio, **kwargs) < retorno:
                baixo = meio
            else:
                alto = meio
        return (baixo + alto) / 2

    def tabela_decisao(self, carteira_base: pd.DataFrame, carteira_estresse: pd.DataFrame) -> pd.DataFrame:
        """Decisao por faixa: o cenario base decide, o de estresse testa a folga.

        - Aprovar: paga o retorno minimo ate no estresse.
        - Aprovar com condicoes: paga no cenario base, mas nao no estresse.
        - Recusar: nao paga o retorno minimo nem no cenario base.
        """
        minimo = self.premissas.retorno_minimo
        linhas = []
        for faixa in sorted(carteira_base["faixa"].unique()):
            base = carteira_base[carteira_base["faixa"] == faixa]
            estresse = carteira_estresse[carteira_estresse["faixa"] == faixa]
            raroc_base, raroc_estresse = self.raroc(base), self.raroc(estresse)
            if raroc_estresse >= minimo:
                decisao = "Aprovar"
            elif raroc_base >= minimo:
                decisao = "Aprovar com condições"
            else:
                decisao = "Recusar"
            linhas.append({
                "faixa": faixa,
                "pd_modelo_%": base["pd_modelo"].mean() * 100,
                "raroc_base_%": raroc_base * 100,
                "raroc_estresse_%": raroc_estresse * 100,
                "taxa_minima_base_%": self.taxa_minima(base) * 100,
                "taxa_minima_estresse_%": self.taxa_minima(estresse) * 100,
                "decisao": decisao,
            })
        return pd.DataFrame(linhas).set_index("faixa")

    def avaliar_politica(self, politica: Politica, carteira: pd.DataFrame) -> dict:
        """Resumo de uma politica num cenario: aprovacao, volume, perda, lucro e RAROC."""
        decidida = politica.aplicar(carteira, self.premissas.taxa_juros_mensal)
        aprovados = decidida[decidida["aprovado"]]
        total = self.agregar(self.calcular(aprovados)).iloc[0]
        return {
            "politica": politica.nome,
            "aprovacao_%": len(aprovados) / len(carteira) * 100,
            "credito_concedido": total["credito"],
            "default_medio_%": total["default_medio_%"],
            "perda_sobre_saldo_%": total["perda_sobre_saldo_%"],
            "lucro": total["lucro"],
            "capital": total["capital"],
            "raroc_%": total["raroc_%"],
            "lucro_economico": total["lucro_economico"],
        }
