from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n, k = X.shape
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan), 0, 0

    dof = n - int(rank)
    if dof <= 0:
        return beta, np.full(k, np.nan), np.full(k, np.nan), int(rank), dof

    resid = y - (X @ beta)
    s2 = float((resid @ resid) / dof)
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = s2 * xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        p_vals = 2.0 * (1.0 - st.t.cdf(np.abs(t_stat), df=dof))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
        p_vals = np.full(k, np.nan)

    return beta, se, p_vals, int(rank), int(dof)


def gerar_decomposicao(
    csv_base: str = "produtos_substitutos_base.csv",
    pasta_saida: str = "saida_substitutos",
    min_obs_sku: int = 60,
) -> Path:
    df = pd.read_csv(csv_base, parse_dates=["data_pedido"])
    req = {"data_pedido", "sku", "quantidade", "qt_disponivel", "desconto", "preco_venda"}
    faltantes = req.difference(set(df.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltantes)}")

    df["sku"] = df["sku"].astype(str)
    df["quantidade"] = df["quantidade"].fillna(0.0)
    df["qt_disponivel"] = df["qt_disponivel"].fillna(0.0)
    df["preco_venda"] = pd.to_numeric(df["preco_venda"], errors="coerce")
    df["desconto"] = pd.to_numeric(df["desconto"], errors="coerce").fillna(0.0)
    df["stockout"] = df["qt_disponivel"] <= 0

    total_dia = df.groupby("data_pedido", as_index=False)["quantidade"].sum().rename(columns={"quantidade": "q_total_dia"})
    total_dia["log_q_total_dia"] = np.log1p(total_dia["q_total_dia"])
    df = df.merge(total_dia[["data_pedido", "log_q_total_dia"]], on="data_pedido", how="left")

    data_ref = df["data_pedido"].min()
    df["t"] = (df["data_pedido"] - data_ref).dt.days.astype(float)
    df["dow"] = df["data_pedido"].dt.dayofweek.astype(int)
    df["month"] = df["data_pedido"].dt.month.astype(int)
    df["log_q"] = np.log1p(df["quantidade"])
    df["log_preco"] = np.where(df["preco_venda"] > 0, np.log(df["preco_venda"]), np.nan)

    base = df[(~df["stockout"]) & (df["preco_venda"] > 0)].copy()

    linhas = []
    for sku, g_train in base.groupby("sku"):
        g_train = g_train.sort_values("data_pedido").copy()
        n = len(g_train)
        if n < min_obs_sku:
            continue
        if g_train["log_preco"].std() < 0.01:
            continue

        y = g_train["log_q"].to_numpy()
        t = g_train["t"].to_numpy()
        t2 = (t / 30.0) ** 2

        cols = [np.ones(n), g_train["log_preco"].to_numpy(), g_train["desconto"].to_numpy(), t, t2, g_train["log_q_total_dia"].to_numpy()]
        for d in range(1, 7):
            cols.append((g_train["dow"].to_numpy() == d).astype(float))
        for m in range(2, 13):
            cols.append((g_train["month"].to_numpy() == m).astype(float))
        X = np.column_stack(cols)

        beta, _, p_vals, rank, dof = _ols(X, y)
        if np.isnan(beta).all():
            continue

        # aplica decomposição para todas as linhas do SKU (inclusive ruptura)
        g_all = df[df["sku"] == sku].copy().sort_values("data_pedido")
        n_all = len(g_all)
        t_all = g_all["t"].to_numpy()
        t2_all = (t_all / 30.0) ** 2
        log_preco_all = g_all["log_preco"].fillna(g_train["log_preco"].median()).to_numpy()
        desconto_all = g_all["desconto"].fillna(0.0).to_numpy()
        org = beta[0] + (beta[3] * t_all) + (beta[4] * t2_all) + (beta[5] * g_all["log_q_total_dia"].fillna(0.0).to_numpy())

        for d in range(1, 7):
            org += beta[5 + d] * (g_all["dow"].to_numpy() == d).astype(float)
        for m in range(2, 13):
            org += beta[11 + (m - 2)] * (g_all["month"].to_numpy() == m).astype(float)

        comp_preco = beta[1] * log_preco_all
        comp_desc = beta[2] * desconto_all
        pred_log = org + comp_preco + comp_desc
        resid_log = g_all["log_q"].to_numpy() - pred_log

        pred_q = np.expm1(pred_log)
        org_q = np.expm1(org)
        preco_q = np.expm1(org + comp_preco) - np.expm1(org)
        desc_q = np.expm1(org + comp_preco + comp_desc) - np.expm1(org + comp_preco)
        resid_q = g_all["quantidade"].to_numpy() - pred_q

        tmp = pd.DataFrame(
            {
                "data_pedido": g_all["data_pedido"].to_numpy(),
                "sku": sku,
                "quantidade": g_all["quantidade"].to_numpy(),
                "stockout": g_all["stockout"].to_numpy(),
                "preco_venda": g_all["preco_venda"].to_numpy(),
                "desconto": g_all["desconto"].to_numpy(),
                "pred_quantidade": pred_q,
                "componente_organico": org_q,
                "componente_preco": preco_q,
                "componente_desconto": desc_q,
                "residual": resid_q,
                "componente_organico_log": org,
                "componente_preco_log": comp_preco,
                "componente_desconto_log": comp_desc,
                "residual_log": resid_log,
                "elasticidade_preco": float(beta[1]),
                "p_value_preco": float(p_vals[1]) if np.isfinite(p_vals[1]) else np.nan,
                "semi_elasticidade_desconto": float(beta[2]),
                "p_value_desconto": float(p_vals[2]) if np.isfinite(p_vals[2]) else np.nan,
                "p_value_t": float(p_vals[3]) if np.isfinite(p_vals[3]) else np.nan,
                "p_value_t2": float(p_vals[4]) if np.isfinite(p_vals[4]) else np.nan,
                "p_value_log_q_total_dia": float(p_vals[5]) if np.isfinite(p_vals[5]) else np.nan,
                "n_obs_modelo": n,
                "rank_modelo": rank,
                "dof_modelo": dof,
            }
        )
        linhas.append(tmp)

    out = Path(pasta_saida)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "decomposicao_sku_dia.csv"
    if linhas:
        final = pd.concat(linhas, ignore_index=True).sort_values(["data_pedido", "sku"])
    else:
        final = pd.DataFrame()
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Linhas decomposição: {len(final)}")
    print(f"Arquivo salvo em: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    gerar_decomposicao()
