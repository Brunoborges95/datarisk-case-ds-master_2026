"""Construcao do target de inadimplencia.

Definicao de mercado
--------------------
Basileia II/III (e a Res. CMN 4.966/2021, que substituiu a 2.682/1999 no
Brasil) considera um tomador em default quando ele esta com 90 ou mais
dias de atraso (DPD = days past due) em uma obrigacao relevante.

Em modelos de concessao (application score) o target costuma ser escrito
como ``EVERxMOBy`` ou ``OVERxMOBy``:

- EVER90MOB12: o contrato chegou a 90+ DPD em ALGUM momento dos 12
  primeiros meses de vida (MOB = months on book).
- OVER60MOB06: o contrato ESTA com 60+ DPD exatamente no fim do 6o mes.

O atraso e sempre medido ate o fim da janela: uma parcela que vence no
mes 11 e e paga no mes 14 nao conta como 90+ DPD dentro de MOB12, porque
no fim da janela ela tinha so ~30 dias de atraso.

Target escolhido: EVER30MOB12
-----------------------------
- Janela de 12 meses: e o horizonte da PD de Basileia.
- Limiar de 30 dias em vez de 90: a carteira tem poucos defaults (395
  EVER90MOB12 em 90 mil contratos). Com 30 dias sao 1.947 maus, e um
  mesmo modelo treinado com EVER30MOB12 ordenou melhor ate o proprio
  EVER90MOB12 em safras futuras do que modelos treinados com EVER60 ou
  EVER90 (secao 6.4 do notebook).
- Custo: so ~36% de quem chega a 30+ DPD evolui para 90+ (``roll_rate``),
  entao o target mistura atraso temporario com default. Por isso
  EVER90MOB12 continua calculado para medir a perda no padrao regulatorio.

Data de observacao
------------------
So rotulamos um contrato quando a janela inteira ja passou, isto e,
``data_decisao + meses <= data_observacao``. Sem essa regra, um contrato
aprovado ha 2 meses entraria como "bom" apenas por nao ter tido tempo de
atrasar.

A data de observacao e POR CLIENTE. Nestas bases, o historico de cada
cliente termina na data do pedido dele na base de submissao (de 04 a
24/02/2025): nenhum cliente tem registro no dia do proprio pedido ou
depois. Usar uma data unica (a ultima data das bases, 2025-02-22)
rotularia contratos cuja janela ainda nao foi observada por inteiro. As
datas chegam em ``observacao_por_cliente``; sem elas, vale a ultima data
presente nas bases para todos.

Limitacoes dos dados tratadas aqui
----------------------------------
1. ``historico_parcelas`` registra praticamente so parcelas PAGAS. Quem
   para de pagar simplesmente some da tabela. Para contratos parcelados
   ainda abertos (sem ``data_encerramento``) e com cronograma incompleto,
   criamos a proxima parcela esperada (ultimo vencimento + 30 dias) sem
   pagamento. Contratos encerrados com menos parcelas que o planejado
   foram quitados antecipadamente e nao recebem parcela faltante.
2. Versoes de parcela: usamos o vencimento mais antigo (vencimento
   original) e o pagamento mais recente (quando a parcela foi quitada).
3. ``data_liberacao`` falta em 96% dos contratos, entao o MOB comeca na
   ``data_decisao``. O primeiro vencimento cai ~31 dias depois (mediana),
   coerente com essa origem.
4. Revolving loans nao tem cronograma (``qtd_parcelas_planejadas = 0``);
   para eles so usamos as parcelas observadas.
"""

from __future__ import annotations

import pandas as pd

# (tipo, dias de atraso, meses de janela). A primeira e o target principal.
DEFINICOES = [
    ("ever", 30, 12),  # principal: mais eventos e melhor ordenacao (secao 6.4)
    ("ever", 60, 12),  # 60+ DPD e o "ponto sem volta": 83% chegam a 90+
    ("ever", 90, 12),  # default de Basileia literal (90+ DPD em 12 meses)
    ("over", 90, 12),  # em default no fim do 12o mes
    ("ever", 30, 6),   # alerta precoce, safras mais recentes
    ("over", 60, 6),   # sugerido no case, na convencao de mercado (status no MOB6)
]
TARGET_PRINCIPAL = "target_ever30mob12"


class TargetEngineering:
    """Constroi targets EVER/OVER para contratos aprovados.

    Exemplo
    -------
    >>> observacao = base_submissao.set_index("id_cliente")["data_solicitacao"]
    >>> te = TargetEngineering(hist_emprestimos, hist_parcelas, observacao)
    >>> te.construir_target(dias_atraso=90, meses=12)            # EVER90MOB12
    >>> te.construir_target(dias_atraso=60, meses=6, tipo="over")  # OVER60MOB06
    """

    def __init__(self, hist_emprestimos: pd.DataFrame, hist_parcelas: pd.DataFrame,
                 observacao_por_cliente: pd.Series | None = None):
        self.contratos = self._preparar_contratos(hist_emprestimos)
        # Ultima data presente nas bases. Vale para o cliente que nao tiver data propria.
        self.data_observacao = max(
            self.contratos["data_decisao"].max(),
            pd.to_datetime(hist_parcelas["data_real_pagamento"]).max(),
        )
        # Ate quando o historico de cada contrato foi observado (ver "Data de observacao" no topo).
        self.contratos["data_observacao"] = self.data_observacao
        if observacao_por_cliente is not None:
            do_cliente = self.contratos["id_cliente"].map(pd.to_datetime(observacao_por_cliente))
            self.contratos["data_observacao"] = do_cliente.fillna(self.data_observacao)
        self.parcelas = self._preparar_parcelas(hist_parcelas)

    @staticmethod
    def _preparar_contratos(hist_emprestimos: pd.DataFrame) -> pd.DataFrame:
        """Mantem apenas aprovados: so eles tem comportamento de pagamento."""
        colunas = [
            "id_contrato", "id_cliente", "tipo_contrato", "data_decisao",
            "data_encerramento", "qtd_parcelas_planejadas",
        ]
        contratos = hist_emprestimos.loc[
            hist_emprestimos["status_contrato"].eq("Approved"), colunas
        ].copy()
        for coluna in ["data_decisao", "data_encerramento"]:
            contratos[coluna] = pd.to_datetime(contratos[coluna], errors="coerce")
        return contratos.dropna(subset=["data_decisao"]).reset_index(drop=True)

    def _preparar_parcelas(self, hist_parcelas: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por contrato/parcela, incluindo parcelas faltantes."""
        parcelas = hist_parcelas.assign(
            data_prevista_pagamento=pd.to_datetime(hist_parcelas["data_prevista_pagamento"]),
            data_real_pagamento=pd.to_datetime(hist_parcelas["data_real_pagamento"]),
        )
        observadas = (
            parcelas.groupby(["id_contrato", "numero_parcela"], as_index=False)
            .agg(
                data_vencimento=("data_prevista_pagamento", "min"),
                data_pagamento=("data_real_pagamento", "max"),
            )
        )
        observadas = observadas[observadas["id_contrato"].isin(self.contratos["id_contrato"])]
        return pd.concat([observadas, self._parcelas_faltantes(observadas)], ignore_index=True)

    def _parcelas_faltantes(self, observadas: pd.DataFrame) -> pd.DataFrame:
        """Proxima parcela esperada de contratos abertos que pararam de pagar."""
        ultima = observadas.groupby("id_contrato").agg(
            ultimo_numero=("numero_parcela", "max"),
            ultimo_vencimento=("data_vencimento", "max"),
        )
        abertos = self.contratos.set_index("id_contrato").join(ultima, how="inner")
        proximo_vencimento = abertos["ultimo_vencimento"] + pd.Timedelta(days=30)
        parou_de_pagar = (
            abertos["data_encerramento"].isna()
            & abertos["qtd_parcelas_planejadas"].gt(0)
            & abertos["ultimo_numero"].lt(abertos["qtd_parcelas_planejadas"])
            & proximo_vencimento.lt(abertos["data_observacao"])
        )
        faltantes = abertos[parou_de_pagar]
        return pd.DataFrame({
            "id_contrato": faltantes.index,
            "numero_parcela": faltantes["ultimo_numero"].to_numpy() + 1,
            "data_vencimento": proximo_vencimento[parou_de_pagar].to_numpy(),
            "data_pagamento": pd.NaT,
        })

    def construir_target(self, dias_atraso: int = 90, meses: int = 12, tipo: str = "ever") -> pd.DataFrame:
        """Rotula cada contrato aprovado com 1 (mau), 0 (bom) ou NA (imaturo).

        tipo="ever": atingiu ``dias_atraso`` em algum momento da janela.
        tipo="over": estava com ``dias_atraso`` ou mais no fim da janela.
        """
        if tipo not in {"ever", "over"}:
            raise ValueError("tipo deve ser 'ever' ou 'over'.")
        nome = f"target_{tipo}{dias_atraso}mob{meses:02d}"

        contratos = self.contratos[["id_contrato", "id_cliente", "tipo_contrato", "data_decisao", "data_observacao"]].copy()
        contratos["fim_janela"] = contratos["data_decisao"] + pd.DateOffset(months=meses)

        parcelas = self.parcelas.merge(contratos[["id_contrato", "fim_janela"]], on="id_contrato")
        parcelas = parcelas[parcelas["data_vencimento"] < parcelas["fim_janela"]]

        # Atraso medido ate o pagamento ou ate o fim da janela, o que vier primeiro.
        data_medicao = parcelas["data_pagamento"].where(
            parcelas["data_pagamento"] < parcelas["fim_janela"], parcelas["fim_janela"]
        )
        parcelas["dpd"] = (data_medicao - parcelas["data_vencimento"]).dt.days.clip(lower=0)
        if tipo == "over":
            em_aberto_no_fim = ~(parcelas["data_pagamento"] < parcelas["fim_janela"])
            parcelas["dpd"] = parcelas["dpd"].where(em_aberto_no_fim, 0)

        mau = parcelas.groupby("id_contrato")["dpd"].max().ge(dias_atraso)
        tem_parcela = contratos["id_contrato"].isin(parcelas["id_contrato"])
        maduro = contratos["fim_janela"] <= contratos["data_observacao"]

        target = contratos["id_contrato"].isin(mau.index[mau]).astype("Int8")
        contratos[nome] = target.where(maduro & tem_parcela)
        return contratos.drop(columns=["fim_janela", "data_observacao"])

    def construir_target_com_indeterminados(self, dias_mau: int = 60, dias_bom: int = 15, meses: int = 12) -> pd.DataFrame:
        """Mau, bom e indeterminado, como em scorecards de mercado.

        - mau (1): atingiu ``dias_mau`` ou mais de atraso na janela
        - bom (0): nunca passou de ``dias_bom`` dias de atraso na janela
        - indeterminado (NA): ficou entre os dois limiares; sai do treino,
          mas continua sendo pontuado em producao
        """
        nome = f"target_mau{dias_mau}_bom{dias_bom}_mob{meses:02d}"
        contratos = self.construir_target(dias_mau, meses)
        mau = contratos.iloc[:, -1]
        passou_do_bom = self.construir_target(dias_bom + 1, meses).iloc[:, -1]
        contratos[nome] = mau.where(mau.eq(1) | passou_do_bom.eq(0))
        return contratos.drop(columns=mau.name)

    def construir_targets(self, definicoes: list[tuple[str, int, int]] = DEFINICOES) -> pd.DataFrame:
        """Uma linha por contrato aprovado com uma coluna por definicao de target."""
        tabela = self.contratos[["id_contrato"]].copy()
        for tipo, dias, meses in definicoes:
            target = self.construir_target(dias, meses, tipo)
            tabela[target.columns[-1]] = target.iloc[:, -1].to_numpy()
        com_indeterminados = self.construir_target_com_indeterminados()
        tabela[com_indeterminados.columns[-1]] = com_indeterminados.iloc[:, -1].to_numpy()
        return tabela

    def resumo_targets(self, definicoes: list[tuple[str, int, int]] = DEFINICOES) -> pd.DataFrame:
        """Compara volume rotulado e taxa de maus entre definicoes de target."""
        targets = self.construir_targets(definicoes).merge(
            self.contratos[["id_contrato", "data_decisao"]], on="id_contrato"
        )
        linhas = []
        for nome in [c for c in targets.columns if c.startswith("target_")]:
            target = targets[nome]
            linhas.append({
                "target": nome,
                "rotulados": int(target.notna().sum()),
                "maus": int(target.eq(1).sum()),
                "taxa_maus_%": round(float(target.mean()) * 100, 2),
                "safra_mais_recente": targets.loc[target.notna(), "data_decisao"].max().date(),
            })
        return pd.DataFrame(linhas).set_index("target")

    def roll_rate(self, limiares: tuple[int, ...] = (5, 15, 30, 45, 60), meses_historia: int = 24) -> pd.DataFrame:
        """Dos contratos que atingiram X dias de atraso, quantos chegaram a 90+?

        Usa contratos com pelo menos ``meses_historia`` de vida para que o
        atraso tenha tempo de evoluir. E a analise classica para escolher o
        limiar do target: o "ponto sem volta" a partir do qual quase todos
        viram default.
        """
        parcelas = self.parcelas.merge(self.contratos[["id_contrato", "data_decisao", "data_observacao"]], on="id_contrato")
        parcelas = parcelas[parcelas["data_decisao"] + pd.DateOffset(months=meses_historia) <= parcelas["data_observacao"]]
        dpd = (parcelas["data_pagamento"].fillna(parcelas["data_observacao"]) - parcelas["data_vencimento"]).dt.days
        dpd_max = dpd.clip(lower=0).groupby(parcelas["id_contrato"]).max()
        return pd.DataFrame([
            {
                "atingiu_dpd": f"{limiar}+",
                "contratos": int((dpd_max >= limiar).sum()),
                "chegaram_a_90+_%": round(float((dpd_max[dpd_max >= limiar] >= 90).mean()) * 100, 1),
            }
            for limiar in limiares
        ]).set_index("atingiu_dpd")

    def curva_safra(self, dias_atraso: int = 90, meses: tuple[int, ...] = (3, 6, 9, 12, 18, 24)) -> pd.Series:
        """Taxa EVER acumulada por MOB, so com safras que ja completaram o maior MOB.

        Mostra quando a inadimplencia para de crescer: a janela do target
        deve ser longa o bastante para capturar a maior parte dos defaults.
        """
        fim_maior_mob = self.contratos["data_decisao"] + pd.DateOffset(months=max(meses))
        completos = self.contratos.loc[fim_maior_mob <= self.contratos["data_observacao"], "id_contrato"]
        taxas = {}
        for mob in meses:
            tabela = self.construir_target(dias_atraso, mob)
            tabela = tabela[tabela["id_contrato"].isin(completos)]
            taxas[mob] = float(tabela.iloc[:, -1].mean()) * 100
        return pd.Series(taxas, name=f"ever{dias_atraso}_%").rename_axis("mob")
