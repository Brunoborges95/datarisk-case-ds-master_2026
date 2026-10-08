"""Visualizacoes reutilizaveis para exploracao de DataFrames."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataviz(
    dataset: pd.DataFrame,
    feature_cat: Sequence[str] | None = None,
    feature_num: Sequence[str] | None = None,
    feature_data: Sequence[str] | None = None,
    *,
    feature_volume: str | None = None,
    feature_data_groupby: str | None = None,
    data_frequency: str = "M",
    volume_aggregation: str = "count",
    max_volume_groups: int = 6,
    min_category_count: int | None = None,
    max_categories: int = 10,
    bins: int = 30,
) -> dict[str, plt.Figure | None]:
    """Cria grids de distribuicoes e de volumetria temporal.

    Parameters
    ----------
    dataset : pandas.DataFrame
        DataFrame a explorar. A funcao nao modifica o objeto recebido.
    feature_cat : sequence[str], optional
        Colunas categoricas/discretas a plotar. Categorias raras sao agrupadas
        em ``Outros``; dados faltantes aparecem em barra propria apenas quando
        existem na coluna.
    feature_num : sequence[str], optional
        Colunas numericas a plotar em histogramas, com linha vertical na mediana.
        O limite superior do grafico e cortado no percentil 99 apenas para
        visualizacao; os valores originais nao sao modificados.
    feature_data : sequence[str], optional
        Colunas de data. Para cada uma, plota o range observado e uma serie de
        volume ao longo do tempo.
    feature_volume : str, optional
        Coluna usada para a volumetria temporal. Com ``volume_aggregation="count"``
        conta registros (valor padrao); com ``"nunique"`` conta valores unicos,
        por exemplo clientes; com ``"sum"`` soma uma medida numerica.
    feature_data_groupby : str, optional
        Coluna categorica para separar a volumetria temporal em series. Grupos
        menos frequentes que os primeiros ``max_volume_groups`` viram ``Outros``.
    data_frequency : {"D", "M"}, default="M"
        Frequencia da volumetria: diaria (``"D"``) ou mensal (``"M"``).
    volume_aggregation : {"count", "nunique", "sum"}, default="count"
        Agregacao aplicada a ``feature_volume`` em cada periodo.
    max_volume_groups : int, default=6
        Numero maximo de series categoricas, alem da serie ``Outros``.
    min_category_count : int, optional
        Frequencia minima para uma categoria aparecer individualmente. Se nao
        informado, usa max(100, 1% do numero de linhas).
    max_categories : int, default=10
        Numero maximo de categorias exibidas individualmente, alem de ``Outros``
        e da barra de faltantes. As mais frequentes sao mantidas.
    bins : int, default=30
        Numero de bins para histogramas numericos.

    Returns
    -------
    dict[str, matplotlib.figure.Figure | None]
        Dicionario com as figuras ``categoricas``, ``numericas`` e ``datas``.
            Uma entrada sera ``None`` quando a lista correspondente estiver vazia.
            As figuras sao devolvidas abertas para permitir ajustes ou salvamento.

    Examples
    --------
    ``dataviz(pedidos, feature_data=["data_solicitacao"],\
    feature_volume="id_cliente", volume_aggregation="nunique",\
    feature_data_groupby="tipo_contrato", data_frequency="M")``
    """
    feature_cat = list(feature_cat or [])
    feature_num = list(feature_num or [])
    feature_data = list(feature_data or [])
    selecionadas = feature_cat + feature_num + feature_data
    colunas_referenciadas = selecionadas.copy()
    if feature_volume is not None:
        colunas_referenciadas.append(feature_volume)
    if feature_data_groupby is not None:
        colunas_referenciadas.append(feature_data_groupby)
    ausentes = sorted(set(colunas_referenciadas).difference(dataset.columns))
    if ausentes:
        raise ValueError(f"Features nao encontradas no dataset: {ausentes}")
    repetidas = sorted(
        coluna for coluna in set(selecionadas) if selecionadas.count(coluna) > 1
    )
    if repetidas:
        raise ValueError(f"Features repetidas entre grupos: {repetidas}")
    if max_categories < 1:
        raise ValueError("max_categories deve ser pelo menos 1.")
    if bins < 1:
        raise ValueError("bins deve ser pelo menos 1.")
    if data_frequency not in {"D", "M", "Y", "A"}:
        raise ValueError("data_frequency deve ser 'D' (diaria), 'M' (mensal), 'Y' (anual) ou 'A' (automática)")
    if volume_aggregation not in {"count", "nunique", "sum"}:
        raise ValueError("volume_aggregation deve ser 'count', 'nunique' ou 'sum'.")
    if volume_aggregation != "count" and feature_volume is None:
        raise ValueError(
            "Informe feature_volume para agregacoes diferentes de 'count'."
        )
    if max_volume_groups < 1:
        raise ValueError("max_volume_groups deve ser pelo menos 1.")
    if volume_aggregation == "sum" and feature_volume is not None:
        if not pd.api.types.is_numeric_dtype(dataset[feature_volume]):
            raise ValueError("feature_volume precisa ser numerica para 'sum'.")

    limiar = (
        max(100, ceil(len(dataset) * 0.01))
        if min_category_count is None
        else min_category_count
    )
    if limiar < 1:
        raise ValueError("min_category_count deve ser positivo.")

    figuras: dict[str, plt.Figure | None] = {
        "categoricas": None,
        "numericas": None,
        "datas": None,
    }

    if feature_cat:
        n_colunas = min(3, len(feature_cat))
        n_linhas = ceil(len(feature_cat) / n_colunas)
        figura, eixos = plt.subplots(
            n_linhas,
            n_colunas,
            figsize=(6 * n_colunas, 4.5 * n_linhas),
            squeeze=False,
        )
        for eixo, coluna in zip(eixos.flat, feature_cat):
            serie = dataset[coluna].replace(
                ["XNA", "Unknown", "UNKNOWN", "N/A", "NA", "None"], np.nan
            )
            frequencias = serie.dropna().astype(str).value_counts()
            faltantes = int(serie.isna().sum())
            frequencias_relevantes = frequencias[frequencias >= limiar].nlargest(
                max_categories
            )
            outras = int(frequencias.sum() - frequencias_relevantes.sum())
            barras = frequencias_relevantes.copy()
            if outras:
                barras["Outros"] = outras
            if faltantes:
                barras["(Dados faltantes)"] = faltantes
            if barras.empty:
                eixo.set_visible(False)
                continue

            cores = [
                "#c44e52" if categoria == "(Dados faltantes)" else "#55a868"
                for categoria in barras.index
            ]
            eixo.bar(barras.index.astype(str), barras.values, color=cores)
            eixo.set_title(f"{coluna} (n={len(dataset):,})")
            eixo.set_ylabel("Registros")
            eixo.tick_params(axis="x", labelrotation=45, labelsize=8)
            for tick in eixo.get_xticklabels():
                tick.set_horizontalalignment("right")

        for eixo in eixos.flat[len(feature_cat) :]:
            eixo.set_visible(False)
        figura.suptitle(
            f"Distribuicao categorica (frequencia minima individual: {limiar})",
            y=1.01,
        )
        figura.tight_layout()
        figuras["categoricas"] = figura

    if feature_num:
        n_colunas = min(3, len(feature_num))
        n_linhas = ceil(len(feature_num) / n_colunas)
        figura, eixos = plt.subplots(
            n_linhas,
            n_colunas,
            figsize=(6 * n_colunas, 4.2 * n_linhas),
            squeeze=False,
        )
        for eixo, coluna in zip(eixos.flat, feature_num):
            valores = pd.to_numeric(dataset[coluna], errors="coerce").dropna()
            if valores.empty:
                eixo.set_visible(False)
                continue
            mediana = float(valores.median())
            limite_visual = float(valores.quantile(0.99))
            eixo.hist(
                valores.clip(upper=limite_visual),
                bins=bins,
                color="#4c72b0",
                alpha=0.85,
            )
            eixo.axvline(
                mediana,
                color="#c44e52",
                linestyle="--",
                linewidth=2,
                label=f"Mediana: {mediana:,.2f}",
            )
            eixo.set_title(f"{coluna} (n={len(valores):,}; p99={limite_visual:,.2f})")
            eixo.set_ylabel("Registros")
            eixo.legend(fontsize=8)

        for eixo in eixos.flat[len(feature_num) :]:
            eixo.set_visible(False)
        figura.suptitle("Distribuicoes numericas", y=1.01)
        figura.tight_layout()
        figuras["numericas"] = figura

    if feature_data:
        # Pega apenas a primeira coluna de data para exibir em um único gráfico
        coluna = feature_data[0] if isinstance(feature_data, (list, tuple)) else feature_data
        
        # Cria uma figura e um eixo únicos
        figura, eixo_volume = plt.subplots(
            figsize=(8, 5),
        )
        
        datas = pd.to_datetime(dataset[coluna], errors="coerce").dropna()
        
        if not datas.empty:
            data_min, data_max = datas.min(), datas.max()

            temporal = dataset.loc[datas.index].copy()
            temporal["_data_plot"] = pd.to_datetime(temporal[coluna], errors="coerce")
            
            if data_frequency == "A":
                intervalo_dias = (data_max - data_min).days

                if intervalo_dias <= 31:
                    frequencia_efetiva = "D"
                elif intervalo_dias <= 730:
                    frequencia_efetiva = "M"
                else:
                    frequencia_efetiva = "Y"
            else:
                frequencia_efetiva = data_frequency

            if frequencia_efetiva == "M":
                temporal["_periodo"] = (
                    temporal["_data_plot"]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
            elif frequencia_efetiva == "Y":
                temporal["_periodo"] = (
                    temporal["_data_plot"]
                    .dt.to_period("Y")
                    .dt.to_timestamp()
                )
            else:
                temporal["_periodo"] = temporal["_data_plot"].dt.normalize()
            data_frequency = frequencia_efetiva 
            temporal = temporal.dropna(subset=["_periodo"])

            # Função auxiliar para evitar a repetição das lógicas de agregação
            def calcular_volume(df_agrupado):
                if volume_aggregation == "count":
                    return df_agrupado.groupby("_periodo").size()
                if volume_aggregation == "nunique":
                    return df_agrupado.groupby("_periodo")[feature_volume].nunique()
                return df_agrupado.groupby("_periodo")[feature_volume].sum(min_count=1)

            if feature_data_groupby is None:
                volume = calcular_volume(temporal)
                eixo_volume.plot(
                    volume.index,
                    volume.values,
                    marker="o" if data_frequency == "M" else None,
                    linewidth=1.5,
                )
            else:
                grupos = temporal[feature_data_groupby].astype("string").fillna("(Dados faltantes)")
                grupos_frequentes = grupos.value_counts().head(max_volume_groups).index
                temporal["_grupo_plot"] = grupos.where(grupos.isin(grupos_frequentes), "Outros")
                
                for nome_grupo, subgrupo in temporal.groupby("_grupo_plot", sort=False):
                    volume = calcular_volume(subgrupo)
                    eixo_volume.plot(
                        volume.index,
                        volume.values,
                        marker="o" if data_frequency == "M" else None,
                        linewidth=1.4,
                        label=str(nome_grupo),
                    )
                eixo_volume.legend(
                    title=feature_data_groupby,
                    fontsize=8,
                    title_fontsize=8,
                )

                ag = {"Y": "ano", "M": "mês", "D": "dia"}.get(data_frequency)

                eixo_volume.set_title(
                    f"{coluna}: {data_min:%Y-%m-%d} a {data_max:%Y-%m-%d}\n"
                    f"{feature_volume} - Volumetria de {ag} "
                    f"(n={len(datas):,}; faltantes={dataset[coluna].isna().sum():,})"
                )
            
            eixo_volume.set_ylabel(
                "Registros" if volume_aggregation == "count"
                else "Valores únicos" if volume_aggregation == "nunique"
                else "Soma"
            )
            
            eixo_volume.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m" if data_frequency == "M" else "%Y-%m-%d")
            )
            eixo_volume.xaxis.set_major_locator(mdates.AutoDateLocator())
            eixo_volume.tick_params(axis="x", rotation=30)
            eixo_volume.grid(axis="both", alpha=0.25)
            
        figura.suptitle("Volumetria Temporal", y=1.01)
        figura.tight_layout()
        figuras["datas"] = figura

    return figuras