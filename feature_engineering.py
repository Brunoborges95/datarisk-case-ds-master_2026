"""Features para o modelo de inadimplencia.

Duas formas de uso:

1. EDA: os metodos ``enriquecer_*`` recebem uma base e devolvem a mesma
   base com colunas derivadas, sem mudar a granularidade.
2. Modelagem: ``construir_base`` recebe uma tabela de pedidos (uma linha
   por pedido) e devolve somente colunas conhecidas no momento do pedido:
   dados do pedido, cadastro e historico do cliente ANTES da data de corte.

A data de corte e ``data_solicitacao`` na base de submissao e
``data_decisao`` no historico (treino). Usar so eventos anteriores ao
corte garante que o modelo treinado no passado veja exatamente o mesmo
tipo de informacao que tera em tempo real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTINELAS = ["XNA", "XAP", "Unknown", "UNKNOWN", "N/A", "NA", "None"]

COLUNAS_PEDIDO = [
    "tipo_contrato", "valor_credito", "valor_bem", "valor_parcela",
    "dia_semana_solicitacao", "hora_solicitacao",
]
COLUNAS_CADASTRO = [
    "sexo", "qtd_filhos", "qtd_membros_familia", "renda_anual", "tipo_renda",
    "ocupacao", "tipo_organizacao", "nivel_educacao", "estado_civil",
    "tipo_moradia", "possui_carro", "possui_imovel", "nota_regiao_cliente",
    "nota_regiao_cliente_cidade",
]


def _razao(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Divisao que devolve NaN quando o denominador e zero, negativo ou nulo."""
    return numerador / denominador.where(denominador > 0)


class FeatureEngineering:
    """Gera features de cadastro, pedido e historico do cliente.

    Exemplo
    -------
    >>> fe = FeatureEngineering(hist_emprestimos, hist_parcelas, base_cadastral)
    >>> base_score = fe.construir_base(base_submissao, "data_solicitacao")
    """

    def __init__(
        self,
        hist_emprestimos: pd.DataFrame,
        hist_parcelas: pd.DataFrame,
        base_cadastral: pd.DataFrame,
    ):
        self.cadastro = self.enriquecer_cadastro(base_cadastral)
        self.pedidos_anteriores = self._preparar_pedidos(hist_emprestimos)
        self.parcelas_anteriores = self._preparar_parcelas(hist_parcelas)

    # ------------------------------------------------------------------
    # Enriquecimento por base (usado na EDA)
    # ------------------------------------------------------------------
    @staticmethod
    def enriquecer_cadastro(base_cadastral: pd.DataFrame) -> pd.DataFrame:
        """Renda per capita e sentinelas (XNA etc.) como nulo."""
        df = base_cadastral.copy()
        df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce")
        df["renda_per_capita"] = _razao(df["renda_anual"], df["qtd_membros_familia"])
        for col in df.select_dtypes(include=["object", "category", "string"]).columns:
            df[col] = df[col].replace(SENTINELAS, np.nan)
        return df

    @staticmethod
    def enriquecer_pedido(base: pd.DataFrame) -> pd.DataFrame:
        """Razoes financeiras do pedido (vale para submissao e historico).

        - parcela_sobre_credito: quanto da divida e paga por mes (~1/prazo)
        - prazo_estimado_meses: valor_credito / valor_parcela
        - credito_sobre_bem: credito acima do bem indica seguro/taxas financiados
        """
        df = base.copy()
        df["parcela_sobre_credito"] = _razao(df["valor_parcela"], df["valor_credito"])
        df["prazo_estimado_meses"] = _razao(df["valor_credito"], df["valor_parcela"])
        df["credito_sobre_bem"] = _razao(df["valor_credito"], df["valor_bem"])
        return df

    @classmethod
    def enriquecer_emprestimos(cls, hist_emprestimos: pd.DataFrame) -> pd.DataFrame:
        """Datas convertidas, razoes do pedido e intervalos do contrato.

        Intervalos como ``dias_decisao_ate_liberacao`` e ``duracao_original_dias``
        so existem depois da aprovacao: servem para EDA, nao como preditores.
        """
        df = cls.enriquecer_pedido(hist_emprestimos)
        for col in [c for c in df.columns if c.startswith("data_")]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["solicitado_sobre_credito"] = _razao(df["valor_solicitado"], df["valor_credito"])
        df["entrada_sobre_bem"] = _razao(df["valor_entrada"], df["valor_bem"])
        df["dias_decisao_ate_liberacao"] = (df["data_liberacao"] - df["data_decisao"]).dt.days
        df["duracao_original_dias"] = (
            df["data_ultimo_vencimento_original"] - df["data_primeiro_vencimento"]
        ).dt.days
        return df

    @staticmethod
    def enriquecer_parcelas(hist_parcelas: pd.DataFrame) -> pd.DataFrame:
        """Atraso por linha de pagamento e classificacao do momento do pagamento."""
        df = hist_parcelas.copy()
        for col in ["data_prevista_pagamento", "data_real_pagamento"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        dias = (df["data_real_pagamento"] - df["data_prevista_pagamento"]).dt.days
        df["dias_entre_vencimento_e_pagamento"] = dias
        df["dias_atraso"] = dias.clip(lower=0)
        df["timing_pagamento"] = np.select(
            [dias < 0, dias == 0, dias > 0],
            ["Antecipado", "No vencimento", "Apos vencimento"],
            default="Sem pagamento",
        )
        df["valor_pago_sobre_previsto"] = _razao(df["valor_pago_parcela"], df["valor_previsto_parcela"])
        return df

    # ------------------------------------------------------------------
    # Preparacao do historico (uma vez, no construtor)
    # ------------------------------------------------------------------
    @staticmethod
    def _preparar_pedidos(hist_emprestimos: pd.DataFrame) -> pd.DataFrame:
        """Pedidos anteriores com flags de status prontas para somar."""
        pedidos = hist_emprestimos[
            ["id_cliente", "data_decisao", "status_contrato", "valor_credito"]
        ].copy()
        pedidos["data_decisao"] = pd.to_datetime(pedidos["data_decisao"], errors="coerce")
        for status, nome in [("Approved", "aprovado"), ("Refused", "recusado"), ("Canceled", "cancelado")]:
            pedidos[nome] = pedidos["status_contrato"].eq(status).astype(int)
        pedidos["valor_aprovado"] = pedidos["valor_credito"].where(pedidos["aprovado"].eq(1))
        return pedidos.dropna(subset=["data_decisao"]).drop(columns="status_contrato")

    @staticmethod
    def _preparar_parcelas(hist_parcelas: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por contrato/parcela: vencimento original e data de quitacao."""
        parcelas = hist_parcelas.assign(
            data_prevista_pagamento=pd.to_datetime(hist_parcelas["data_prevista_pagamento"]),
            data_real_pagamento=pd.to_datetime(hist_parcelas["data_real_pagamento"]),
        )
        return (
            parcelas.groupby(["id_cliente", "id_contrato", "numero_parcela"], as_index=False)
            .agg(
                vencimento=("data_prevista_pagamento", "min"),
                pagamento=("data_real_pagamento", "max"),
            )
            .dropna(subset=["vencimento"])
        )

    # ------------------------------------------------------------------
    # Features historicas point-in-time
    # ------------------------------------------------------------------
    def _features_pedidos(self, cortes: pd.DataFrame) -> pd.DataFrame:
        """Quantos pedidos o cliente fez antes do corte e como terminaram."""
        m = cortes.merge(self.pedidos_anteriores, on="id_cliente")
        m = m[m["data_decisao"] < m["data_corte"]]
        dias_atras = (m["data_corte"] - m["data_decisao"]).dt.days
        m["ultimos_90d"] = (dias_atras <= 90).astype(int)
        m["ultimos_365d"] = (dias_atras <= 365).astype(int)
        m["dias_atras"] = dias_atras

        f = m.groupby("linha").agg(
            hist_n_pedidos=("aprovado", "size"),
            hist_n_aprovados=("aprovado", "sum"),
            hist_n_recusados=("recusado", "sum"),
            hist_n_cancelados=("cancelado", "sum"),
            hist_n_pedidos_90d=("ultimos_90d", "sum"),
            hist_n_pedidos_365d=("ultimos_365d", "sum"),
            hist_valor_aprovado_medio=("valor_aprovado", "mean"),
            hist_dias_desde_ultimo_pedido=("dias_atras", "min"),
        )
        f["hist_taxa_recusa"] = f["hist_n_recusados"] / f["hist_n_pedidos"]
        return f

    def _features_parcelas(self, cortes: pd.DataFrame) -> pd.DataFrame:
        """Comportamento de pagamento nas parcelas que venceram antes do corte.

        Uma parcela vencida antes do corte pode estar:
        - paga antes do corte: atraso = pagamento - vencimento
        - ainda em aberto no corte: o cliente esta inadimplente naquele dia
        """
        m = cortes.merge(self.parcelas_anteriores, on="id_cliente")
        m = m[m["vencimento"] < m["data_corte"]]

        paga_antes = m["pagamento"] < m["data_corte"]
        data_medicao = m["pagamento"].where(paga_antes, m["data_corte"])
        m["dpd"] = (data_medicao - m["vencimento"]).dt.days.clip(lower=0)
        m["em_aberto"] = (~paga_antes).astype(int)
        m["dpd_em_aberto"] = m["dpd"].where(~paga_antes, 0)
        m["atraso"] = (m["dpd"] > 0).astype(int)
        m["atraso_30"] = (m["dpd"] >= 30).astype(int)
        m["atraso_60"] = (m["dpd"] >= 60).astype(int)
        m["ultimos_365d"] = ((m["data_corte"] - m["vencimento"]).dt.days <= 365).astype(int)
        m["atraso_365d"] = m["atraso"] * m["ultimos_365d"]
        m["dias_desde_atraso"] = (m["data_corte"] - m["vencimento"]).dt.days.where(m["atraso"].eq(1))

        f = m.groupby("linha").agg(
            hist_n_parcelas=("dpd", "size"),
            hist_n_contratos_com_parcela=("id_contrato", "nunique"),
            hist_n_parcelas_atraso=("atraso", "sum"),
            hist_n_parcelas_atraso_30=("atraso_30", "sum"),
            hist_n_parcelas_atraso_60=("atraso_60", "sum"),
            hist_dpd_max=("dpd", "max"),
            hist_dpd_medio=("dpd", "mean"),
            hist_n_parcelas_365d=("ultimos_365d", "sum"),
            hist_n_parcelas_atraso_365d=("atraso_365d", "sum"),
            hist_dias_desde_ultimo_atraso=("dias_desde_atraso", "min"),
            hist_n_parcelas_em_aberto=("em_aberto", "sum"),
            hist_dpd_atual=("dpd_em_aberto", "max"),
        )
        f["hist_taxa_atraso"] = f["hist_n_parcelas_atraso"] / f["hist_n_parcelas"]
        f["hist_taxa_atraso_365d"] = f["hist_n_parcelas_atraso_365d"] / f["hist_n_parcelas_365d"]
        return f

    # ------------------------------------------------------------------
    # Base final
    # ------------------------------------------------------------------
    def construir_base(self, pedidos: pd.DataFrame, coluna_data_corte: str) -> pd.DataFrame:
        """Uma linha por pedido com todas as features conhecidas na data de corte.

        Colunas de identificacao (``id_*``) e targets (``target_*``) presentes
        em ``pedidos`` sao mantidas; qualquer outra coluna pos-decisao do
        historico (liberacao, encerramento, etc.) fica de fora.
        """
        manter = [c for c in pedidos.columns if c.startswith(("id_", "target_"))]
        base = pedidos[manter + COLUNAS_PEDIDO].copy()
        base["data_referencia"] = pd.to_datetime(pedidos[coluna_data_corte], errors="coerce")
        base = self.enriquecer_pedido(base)

        cadastro = self.cadastro[["id_cliente", "data_nascimento", "renda_per_capita"] + COLUNAS_CADASTRO]
        base = base.merge(cadastro, on="id_cliente", how="left")
        base["idade"] = (base["data_referencia"] - base["data_nascimento"]).dt.days // 365.25
        # Divide pela renda declarada sem converter para mensal: a parcela mediana
        # e ~2x renda_anual/12, entao a unidade de renda_anual nao e confiavel.
        base["parcela_sobre_renda"] = _razao(base["valor_parcela"], base["renda_anual"])
        base["credito_sobre_renda"] = _razao(base["valor_credito"], base["renda_anual"])
        base = base.drop(columns="data_nascimento")

        cortes = pd.DataFrame({
            "linha": np.arange(len(base)),
            "id_cliente": base["id_cliente"].to_numpy(),
            "data_corte": base["data_referencia"].to_numpy(),
        })
        historico = self._features_pedidos(cortes).join(self._features_parcelas(cortes), how="outer")
        base = base.join(historico.reindex(np.arange(len(base))).set_axis(base.index))

        contagens = [c for c in base.columns if c.startswith("hist_n_")]
        base[contagens] = base[contagens].fillna(0).astype(int)
        base["hist_sem_historico"] = base["hist_n_pedidos"].eq(0).astype(int)
        return base
