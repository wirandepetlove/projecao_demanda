from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st


def _safe_lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    if X.shape[0] <= X.shape[1]:
        return np.full(X.shape[1], np.nan), 0
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta, int(rank)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), 0


def _ols_inference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    beta, rank = _safe_lstsq(X, y)
    if np.isnan(beta).all():
        k = X.shape[1]
        return beta, np.full(k, np.nan), np.full(k, np.nan), 0

    n, k = X.shape
    dof = n - rank
    if dof <= 0:
        return beta, np.full(k, np.nan), np.full(k, np.nan), int(dof)

    y_hat = X @ beta
    resid = y - y_hat
    s2 = float((resid @ resid) / dof)
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        var_beta = s2 * xtx_inv
        se = np.sqrt(np.clip(np.diag(var_beta), 0, np.inf))
        t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        p_vals = 2.0 * (1.0 - st.t.cdf(np.abs(t_stat), df=dof))
        return beta, se, p_vals, int(dof)
    except np.linalg.LinAlgError:
        return beta, np.full(k, np.nan), np.full(k, np.nan), int(dof)


def estimar_elasticidades(
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
    df["preco_venda"] = df["preco_venda"].astype(float)
    df["desconto"] = df["desconto"].astype(float)
    df["stockout"] = df["qt_disponivel"] <= 0

    # Proxy de crescimento organico: demanda total diaria da categoria.
    total_dia = df.groupby("data_pedido", as_index=False)["quantidade"].sum().rename(columns={"quantidade": "q_total_dia"})
    total_dia["log_q_total_dia"] = np.log1p(total_dia["q_total_dia"])
    df = df.merge(total_dia[["data_pedido", "log_q_total_dia"]], on="data_pedido", how="left")

    # Features de calendario para controlar sazonalidade/expansao.
    data_ref = df["data_pedido"].min()
    df["t"] = (df["data_pedido"] - data_ref).dt.days.astype(float)
    df["dow"] = df["data_pedido"].dt.dayofweek.astype(int)
    df["month"] = df["data_pedido"].dt.month.astype(int)

    # Estimação em dias sem stockout para reduzir viés de censura por ruptura.
    base = df[(~df["stockout"]) & (df["preco_venda"] > 0)].copy()
    base["log_q"] = np.log1p(base["quantidade"])
    base["log_preco"] = np.log(base["preco_venda"])
    base["desc_rate"] = (-base["desconto"]).clip(lower=0.0, upper=0.9)
    base["desc_0_10"] = np.minimum(base["desc_rate"], 0.10)
    base["desc_10_30"] = np.minimum(np.maximum(base["desc_rate"] - 0.10, 0.0), 0.20)
    base["desc_30_plus"] = np.maximum(base["desc_rate"] - 0.30, 0.0)

    resultados = []
    for sku, g in base.groupby("sku"):
        g = g.sort_values("data_pedido").copy()
        n = len(g)
        if n < min_obs_sku:
            continue

        # Variação minima para identificar elasticidade.
        if g["log_preco"].std() < 0.01:
            continue

        # OLS por SKU:
        # log_q ~ log_preco + desconto + t + t2 + log_q_total_dia + dummies_dow + dummies_month
        y = g["log_q"].to_numpy()
        t = g["t"].to_numpy()
        t2 = (t / 30.0) ** 2
        cols = [
            np.ones(n),
            g["log_preco"].to_numpy(),
            g["desconto"].fillna(0.0).to_numpy(),
            t,
            t2,
            g["log_q_total_dia"].fillna(0.0).to_numpy(),
        ]

        for d in range(1, 7):
            cols.append((g["dow"].to_numpy() == d).astype(float))
        for m in range(2, 13):
            cols.append((g["month"].to_numpy() == m).astype(float))

        X = np.column_stack(cols)
        rank = int(np.linalg.matrix_rank(X))
        beta, _, p_vals, dof = _ols_inference(X, y)
        if np.isnan(beta).all():
            continue

        # Indices: 0 const, 1 log_preco, 2 desconto, ...
        elast_preco = float(beta[1])
        semi_elast_desc = float(beta[2])
        p_preco = float(p_vals[1]) if np.isfinite(p_vals[1]) else np.nan
        p_desc = float(p_vals[2]) if np.isfinite(p_vals[2]) else np.nan

        # Modelo não linear de desconto (piecewise) com os mesmos controles.
        cols_nl = [
            np.ones(n),
            g["log_preco"].to_numpy(),
            g["desc_0_10"].to_numpy(),
            g["desc_10_30"].to_numpy(),
            g["desc_30_plus"].to_numpy(),
            t,
            t2,
            g["log_q_total_dia"].fillna(0.0).to_numpy(),
        ]
        for d in range(1, 7):
            cols_nl.append((g["dow"].to_numpy() == d).astype(float))
        for m in range(2, 13):
            cols_nl.append((g["month"].to_numpy() == m).astype(float))
        X_nl = np.column_stack(cols_nl)
        beta_nl, _, p_nl, dof_nl = _ols_inference(X_nl, y)
        if np.isnan(beta_nl).all():
            b_d0_10 = np.nan
            b_d10_30 = np.nan
            b_d30 = np.nan
            p_d0_10 = np.nan
            p_d10_30 = np.nan
            p_d30 = np.nan
        else:
            b_d0_10 = float(beta_nl[2])
            b_d10_30 = float(beta_nl[3])
            b_d30 = float(beta_nl[4])
            p_d0_10 = float(p_nl[2]) if np.isfinite(p_nl[2]) else np.nan
            p_d10_30 = float(p_nl[3]) if np.isfinite(p_nl[3]) else np.nan
            p_d30 = float(p_nl[4]) if np.isfinite(p_nl[4]) else np.nan

        std_log_preco = float(g["log_preco"].std())
        pass_obs = n >= 120
        pass_var_preco = std_log_preco >= 0.03
        pass_p_preco = bool(np.isfinite(p_preco) and p_preco < 0.05)
        pass_range_preco = bool(-10.0 <= elast_preco <= 2.0)
        quality_score = int(pass_obs) + int(pass_var_preco) + int(pass_p_preco) + int(pass_range_preco)

        resultados.append(
            {
                "sku": sku,
                "n_obs": n,
                "rank_modelo": rank,
                "dof_modelo": dof,
                "elasticidade_preco": elast_preco,
                "p_value_preco": p_preco,
                "semi_elasticidade_desconto": semi_elast_desc,
                "p_value_desconto": p_desc,
                "semi_elast_desc_0_10": b_d0_10,
                "p_value_desc_0_10": p_d0_10,
                "semi_elast_desc_10_30": b_d10_30,
                "p_value_desc_10_30": p_d10_30,
                "semi_elast_desc_30_plus": b_d30,
                "p_value_desc_30_plus": p_d30,
                "preco_medio": float(g["preco_venda"].mean()),
                "desconto_medio": float(g["desconto"].fillna(0.0).mean()),
                "q_media": float(g["quantidade"].mean()),
                "share_dias_stockout_base": float(df[df["sku"] == sku]["stockout"].mean()),
                "std_log_preco": std_log_preco,
                "pass_obs": pass_obs,
                "pass_var_preco": pass_var_preco,
                "pass_p_preco": pass_p_preco,
                "pass_range_preco": pass_range_preco,
                "quality_score": quality_score,
            }
        )

    out = Path(pasta_saida)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "elasticidade_sku.csv"
    res = pd.DataFrame(resultados).sort_values("n_obs", ascending=False)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not res.empty:
        curada = res[(res["quality_score"] >= 3) & (res["p_value_preco"] < 0.10)].copy()
        curada.to_csv(out / "elasticidade_sku_curada.csv", index=False, encoding="utf-8-sig")

    print(f"SKUs estimados: {len(res)}")
    print(f"Arquivo salvo em: {out_path.resolve()}")
    if not res.empty:
        print(f"SKUs curados: {len(curada)}")
    if not res.empty:
        print("Resumo elasticidade_preco:")
        print(res["elasticidade_preco"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    return out_path


if __name__ == "__main__":
    estimar_elasticidades()
