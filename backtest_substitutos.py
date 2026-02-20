from pathlib import Path

import numpy as np
import pandas as pd

from modelo_substitutos import ModeloSubstitutos


def calcular_metricas(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else np.nan
    bias = float(np.mean(err))
    return {"mae": mae, "rmse": rmse, "wape": wape, "bias": bias}


def projetar_base(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    train = train_df.copy()
    test = test_df.copy()

    train["dow"] = train["data_pedido"].dt.dayofweek
    test["dow"] = test["data_pedido"].dt.dayofweek

    train_ok = train[train["qt_disponivel"] > 0]
    media_dow = train_ok.groupby(["sku", "dow"])["quantidade"].mean()
    media_sku = train_ok.groupby("sku")["quantidade"].mean()

    def _pred(row: pd.Series) -> float:
        chave = (row["sku"], row["dow"])
        if chave in media_dow.index:
            return float(media_dow.loc[chave])
        if row["sku"] in media_sku.index:
            return float(media_sku.loc[row["sku"]])
        return 0.0

    return test.apply(_pred, axis=1)


def _preparar_arestas(edges: pd.DataFrame, pval_max: float = 0.05) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()

    use = edges[edges["p_value_ajustado"] <= pval_max].copy()
    if use.empty:
        return use

    p = use["p_value_ajustado"].clip(lower=1e-12)
    use["p_score"] = -np.log10(p)
    use["dias_score"] = (use["dias_stockout_anchor"] / 30.0).clip(lower=0.0, upper=1.0)
    use["effect_pos"] = use["effect_d"].clip(lower=0.0)
    use["uplift_pos"] = use["uplift_pct"].clip(lower=0.0)
    use["edge_strength"] = use["effect_pos"] * (1.0 + use["p_score"]) * (0.5 + 0.5 * use["dias_score"])

    # Shrinkage: reduz uplifts instaveis com pouca evidencia.
    dias_conf = use["dias_stockout_anchor"] / (use["dias_stockout_anchor"] + 20.0)
    effect_conf = use["effect_pos"] / (use["effect_pos"] + 0.8)
    p_conf = use["p_score"] / (use["p_score"] + 2.0)
    use["shrink_factor"] = (dias_conf * effect_conf * p_conf).clip(lower=0.0, upper=1.0)
    use["uplift_shrunk"] = use["uplift_pos"] * use["shrink_factor"]
    return use


def _calcular_loss_rate_por_anchor(train_df: pd.DataFrame, anchors: set[str]) -> dict[str, float]:
    base = train_df.copy()
    base["stockout"] = base["qt_disponivel"] <= 0
    taxas: dict[str, float] = {}
    for a in anchors:
        s = base[base["sku"] == a]
        if s.empty:
            taxas[a] = 0.30
            continue
        in_days = s.loc[~s["stockout"], "quantidade"]
        out_days = s.loc[s["stockout"], "quantidade"]
        if len(in_days) < 10 or len(out_days) < 3:
            taxas[a] = 0.30
            continue
        m_in = float(in_days.mean())
        m_out = float(out_days.mean())
        if m_in <= 0:
            taxas[a] = 0.30
            continue
        loss = max(0.0, min(1.0, (m_in - m_out) / m_in))
        taxas[a] = loss
    return taxas


def aplicar_uplift_ponderado(
    test_df: pd.DataFrame,
    pred_base: pd.Series,
    edges_use: pd.DataFrame,
    loss_rate_por_anchor: dict[str, float],
    alpha_por_sku: dict[str, float] | None = None,
    denylist_skus: set[str] | None = None,
    top_k_anchors_por_candidato: int = 2,
    usar_gatilho_perda_real_anchor: bool = True,
    min_ganho_abs_sku_dia: float = 5.0,
    min_ganho_pct_base_sku_dia: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    work = test_df.copy()
    work["pred_base"] = pred_base.values
    work["stockout"] = work["qt_disponivel"] <= 0

    if edges_use.empty:
        return work["pred_base"].copy(), pd.Series(1.0, index=work.index)
    if alpha_por_sku is None:
        alpha_por_sku = {}
    if denylist_skus is None:
        denylist_skus = set()

    # Lookup active edges by anchor
    by_anchor: dict[str, list[tuple[str, float, float]]] = {}
    for r in edges_use.itertuples(index=False):
        by_anchor.setdefault(str(r.sku_anchor), []).append(
            (str(r.sku_candidato), float(r.uplift_shrunk), float(r.edge_strength))
        )

    pred = work["pred_base"].copy()
    mult = pd.Series(1.0, index=work.index)

    for _, g in work.groupby("data_pedido", sort=True):
        anchors_ruptura = set(g.loc[g["stockout"], "sku"].astype(str).tolist())
        idx_por_sku = {str(r["sku"]): idx for idx, r in g.iterrows()}
        base_por_sku = {str(r["sku"]): float(r["pred_base"]) for _, r in g.iterrows()}
        contribs_por_sku: dict[str, list[float]] = {}

        for a in anchors_ruptura:
            if a not in by_anchor:
                continue
            idx_a = idx_por_sku.get(a)
            if idx_a is None:
                continue

            base_anchor = float(pred.at[idx_a])
            qtd_anchor_real = float(g.loc[idx_a, "quantidade"]) if "quantidade" in g.columns else base_anchor
            loss_rate = float(loss_rate_por_anchor.get(a, 0.30))
            budget_anchor = max(0.0, base_anchor * loss_rate)
            if usar_gatilho_perda_real_anchor:
                perda_real_anchor = max(0.0, base_anchor - qtd_anchor_real)
                if perda_real_anchor <= 0:
                    continue
                budget_anchor = min(budget_anchor, perda_real_anchor)
            if budget_anchor <= 0:
                continue

            potenciais: list[tuple[str, float]] = []
            soma_potencial = 0.0
            for cand, uplift, strength in by_anchor[a]:
                base_cand = max(0.0, float(base_por_sku.get(cand, 0.0)))
                potencial = base_cand * max(0.0, uplift) * max(0.0, strength)
                if potencial <= 0:
                    continue
                potenciais.append((cand, potencial))
                soma_potencial += potencial

            if soma_potencial <= 0:
                continue

            escala = min(1.0, budget_anchor / soma_potencial)
            for cand, potencial in potenciais:
                contribs_por_sku.setdefault(cand, []).append(potencial * escala)

        for idx, r in g.iterrows():
            sku = str(r["sku"])
            base = float(r["pred_base"])
            gains = contribs_por_sku.get(sku, [])
            if len(gains) > top_k_anchors_por_candidato:
                gains = sorted(gains, reverse=True)[:top_k_anchors_por_candidato]
            ganho_bruto = float(sum(gains))
            alpha = float(alpha_por_sku.get(sku, 1.0))
            if sku in denylist_skus:
                alpha = 0.0
            alpha = max(0.0, min(1.0, alpha))
            ganho = ganho_bruto * alpha
            ganho_minimo = max(float(min_ganho_abs_sku_dia), float(min_ganho_pct_base_sku_dia) * base)
            if ganho < ganho_minimo:
                ganho = 0.0
            pred.at[idx] = base + ganho
            if base > 0:
                m = 1.0 + (ganho / base)
            else:
                m = 1.0
            mult.at[idx] = m

    return pred, mult


def calibrar_alpha_e_denylist(
    treino: pd.DataFrame,
    pred_base_treino: pd.Series,
    edges_use: pd.DataFrame,
    loss_rate_por_anchor: dict[str, float],
) -> tuple[dict[str, float], set[str], pd.DataFrame]:
    pred_uplift_treino, mult_treino = aplicar_uplift_ponderado(
        treino,
        pred_base_treino,
        edges_use,
        loss_rate_por_anchor,
        alpha_por_sku={},
        denylist_skus=set(),
        top_k_anchors_por_candidato=2,
        usar_gatilho_perda_real_anchor=True,
        min_ganho_abs_sku_dia=5.0,
        min_ganho_pct_base_sku_dia=0.02,
    )

    calib = treino.copy()
    calib["pred_base"] = pred_base_treino.values
    calib["pred_uplift"] = pred_uplift_treino.values
    calib["uplift_aplicado"] = mult_treino.values > 1

    sku_tbl = (
        calib[calib["uplift_aplicado"]]
        .groupby("sku")
        .agg(
            linhas_uplift=("uplift_aplicado", "sum"),
            real_total=("quantidade", "sum"),
            base_total=("pred_base", "sum"),
            uplift_total=("pred_uplift", "sum"),
        )
        .reset_index()
    )
    if sku_tbl.empty:
        return {}, set(), sku_tbl

    sku_tbl["erro_base"] = (sku_tbl["real_total"] - sku_tbl["base_total"]).abs()
    sku_tbl["erro_uplift"] = (sku_tbl["real_total"] - sku_tbl["uplift_total"]).abs()
    sku_tbl["delta_erro"] = sku_tbl["erro_uplift"] - sku_tbl["erro_base"]
    sku_tbl["delta_pct"] = np.where(
        sku_tbl["erro_base"] > 0, sku_tbl["delta_erro"] / sku_tbl["erro_base"], 0.0
    )

    alpha_por_sku: dict[str, float] = {}
    for r in sku_tbl.itertuples(index=False):
        if r.erro_base <= 0:
            alpha = 1.0
        else:
            # 1.0 se melhorou; cai linearmente quando piora.
            alpha = 1.0 - max(0.0, float(r.delta_pct))
        alpha_por_sku[str(r.sku)] = float(max(0.0, min(1.0, alpha)))

    # Piores casos recorrentes ficam em denylist.
    deny = set(
        sku_tbl[
            (sku_tbl["linhas_uplift"] >= 10)
            & (sku_tbl["delta_pct"] > 0.50)
            & (sku_tbl["delta_erro"] > 500.0)
        ]["sku"].astype(str)
    )

    return alpha_por_sku, deny, sku_tbl


def executar_backtest(
    csv_base: str = "produtos_substitutos_base.csv",
    cutoff_treino: str = "2025-12-31",
    inicio_teste: str = "2026-01-01",
    fim_teste: str = "2026-01-31",
    pasta_saida: str = "saida_substitutos",
) -> None:
    base = pd.read_csv(csv_base, parse_dates=["data_pedido"])
    base["sku"] = base["sku"].astype(str)
    base["quantidade"] = base["quantidade"].fillna(0.0)
    base["qt_disponivel"] = base["qt_disponivel"].fillna(0.0)

    cutoff = pd.Timestamp(cutoff_treino)
    ini = pd.Timestamp(inicio_teste)
    fim = pd.Timestamp(fim_teste)

    treino = base[base["data_pedido"] <= cutoff].copy()
    teste = base[(base["data_pedido"] >= ini) & (base["data_pedido"] <= fim)].copy()

    tmp_treino = Path("_tmp_treino_substitutos.csv")
    treino.to_csv(tmp_treino, index=False, encoding="utf-8-sig")

    modelo = ModeloSubstitutos(str(tmp_treino))
    edges = modelo.detectar_arestas()
    edges_use = _preparar_arestas(edges, pval_max=0.05)
    anchors = set(edges_use["sku_anchor"].astype(str).tolist()) if not edges_use.empty else set()
    loss_rate_por_anchor = _calcular_loss_rate_por_anchor(treino, anchors)

    pred_base = projetar_base(treino, teste)
    pred_base_treino = projetar_base(treino, treino)
    alpha_por_sku, denylist_skus, calib_skus = calibrar_alpha_e_denylist(
        treino, pred_base_treino, edges_use, loss_rate_por_anchor
    )
    pred_uplift4, mult_uplift4 = aplicar_uplift_ponderado(
        teste,
        pred_base,
        edges_use,
        loss_rate_por_anchor,
        alpha_por_sku=alpha_por_sku,
        denylist_skus=denylist_skus,
        top_k_anchors_por_candidato=2,
        usar_gatilho_perda_real_anchor=True,
        min_ganho_abs_sku_dia=5.0,
        min_ganho_pct_base_sku_dia=0.02,
    )

    teste = teste.copy()
    teste["stockout"] = teste["qt_disponivel"] <= 0
    teste["pred_base"] = pred_base.values
    teste["pred_uplift_4_shrink"] = pred_uplift4.values
    teste["uplift_mult_4_shrink"] = mult_uplift4.values
    teste["alpha_sku"] = teste["sku"].astype(str).map(alpha_por_sku).fillna(1.0)
    teste["denylist_sku"] = teste["sku"].astype(str).isin(denylist_skus)

    m_base = calcular_metricas(teste["quantidade"], teste["pred_base"])
    m_4 = calcular_metricas(teste["quantidade"], teste["pred_uplift_4_shrink"])

    diario = (
        teste.groupby("data_pedido")[["quantidade", "pred_base", "pred_uplift_4_shrink"]]
        .sum()
        .reset_index()
        .sort_values("data_pedido")
    )
    d_base = calcular_metricas(diario["quantidade"], diario["pred_base"])
    d_4 = calcular_metricas(diario["quantidade"], diario["pred_uplift_4_shrink"])

    out = Path(pasta_saida)
    out.mkdir(parents=True, exist_ok=True)
    teste.to_csv(out / "backtest_jan26_previsoes.csv", index=False, encoding="utf-8-sig")
    diario.to_csv(out / "backtest_jan26_diario.csv", index=False, encoding="utf-8-sig")
    calib_skus.to_csv(out / "backtest_calibracao_sku.csv", index=False, encoding="utf-8-sig")

    resumo = pd.DataFrame(
        [
            {"escopo": "sku_dia", "modelo": "base", **m_base},
            {"escopo": "sku_dia", "modelo": "uplift_4_shrink", **m_4},
            {"escopo": "total_dia", "modelo": "base", **d_base},
            {"escopo": "total_dia", "modelo": "uplift_4_shrink", **d_4},
            {
                "escopo": "meta",
                "modelo": "info",
                "mae": len(edges),
                "rmse": len(edges_use),
                "wape": len(teste),
                "bias": int(teste["stockout"].sum()),
            },
        ]
    )
    resumo.to_csv(out / "backtest_jan26_metricas.csv", index=False, encoding="utf-8-sig")

    if tmp_treino.exists():
        tmp_treino.unlink()

    print(f"Treino ate: {cutoff.date()} | Teste: {ini.date()} a {fim.date()}")
    print(f"Arestas detectadas no treino: {len(edges)}")
    print(f"Arestas usadas (p_value_ajustado <= 0.05): {len(edges_use)}")
    print(f"SKUs calibrados com alpha: {len(alpha_por_sku)} | denylist: {len(denylist_skus)}")
    print("Metricas SKU-dia")
    print(f"Base               | MAE={m_base['mae']:.3f} RMSE={m_base['rmse']:.3f} WAPE={m_base['wape']:.4f}")
    print(f"Uplift 4 shrink    | MAE={m_4['mae']:.3f} RMSE={m_4['rmse']:.3f} WAPE={m_4['wape']:.4f}")
    print("Metricas Total-dia")
    print(f"Base               | MAE={d_base['mae']:.3f} RMSE={d_base['rmse']:.3f} WAPE={d_base['wape']:.4f}")
    print(f"Uplift 4 shrink    | MAE={d_4['mae']:.3f} RMSE={d_4['rmse']:.3f} WAPE={d_4['wape']:.4f}")
    print(f"Arquivos salvos em: {out.resolve()}")


if __name__ == "__main__":
    executar_backtest()
