from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


class ModeloSubstitutos:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = self._carregar_base(self.csv_path)
        self.sales = self._pivot_vendas(self.df)
        self.preco = self._pivot_preco(self.df)
        self.stock = self._pivot_stockout(self.df)
        self.signal = self._sinal_ajustado_por_preco(self.sales, self.preco)
        self.edges = pd.DataFrame()
        self.pares = pd.DataFrame()
        self.grupos = pd.DataFrame()

    @staticmethod
    def _carregar_base(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["data_pedido"])
        df["sku"] = df["sku"].astype(str)
        df["quantidade"] = df["quantidade"].fillna(0.0)
        df["qt_disponivel"] = df["qt_disponivel"].fillna(0.0)
        df["stockout"] = df["qt_disponivel"] <= 0
        return df

    @staticmethod
    def _pivot_vendas(df: pd.DataFrame) -> pd.DataFrame:
        return df.pivot_table(
            index="data_pedido",
            columns="sku",
            values="quantidade",
            aggfunc="sum",
            fill_value=0.0,
        )

    @staticmethod
    def _pivot_stockout(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pivot_table(
                index="data_pedido",
                columns="sku",
                values="stockout",
                aggfunc="max",
                fill_value=False,
            )
            .astype(bool)
        )

    @staticmethod
    def _pivot_preco(df: pd.DataFrame) -> pd.DataFrame:
        if "preco_venda" not in df.columns:
            return pd.DataFrame(index=df["data_pedido"].sort_values().unique())
        return df.pivot_table(
            index="data_pedido",
            columns="sku",
            values="preco_venda",
            aggfunc="mean",
        )

    @staticmethod
    def _residual_por_dia_semana(mat: pd.DataFrame) -> pd.DataFrame:
        resid = mat.copy()
        dow = pd.Series(mat.index.dayofweek, index=mat.index)
        for d in range(7):
            mask = dow == d
            resid.loc[mask] = mat.loc[mask] - mat.loc[mask].mean(axis=0)
        return resid

    @staticmethod
    def _ajustar_fdr_bh(p_values: pd.Series) -> pd.Series:
        p = p_values.astype(float).to_numpy()
        out = np.full_like(p, np.nan, dtype=float)

        valid = ~np.isnan(p)
        if valid.sum() == 0:
            return pd.Series(out, index=p_values.index)

        pv = p[valid]
        m = len(pv)
        order = np.argsort(pv)
        ranked = pv[order]

        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        out[valid] = restored
        return pd.Series(out, index=p_values.index)

    def _sinal_ajustado_por_preco(self, sales: pd.DataFrame, preco: pd.DataFrame) -> pd.DataFrame:
        # Caso não exista preco_venda na base, cai no comportamento antigo (ajuste só por dia da semana).
        if preco.empty:
            return self._residual_por_dia_semana(sales)

        idx = sales.index
        cols = sales.columns
        px = preco.reindex(index=idx, columns=cols)

        # Preenche preço faltante por mediana do sku e depois mediana global.
        px = px.apply(lambda c: c.fillna(c.median()), axis=0)
        global_med = float(np.nanmedian(px.to_numpy()))
        if np.isnan(global_med):
            global_med = 1.0
        px = px.fillna(global_med)
        px = px.clip(lower=1e-6)

        y_log = np.log1p(sales)
        x_log = np.log(px)

        y_resid = self._residual_por_dia_semana(y_log)
        x_resid = self._residual_por_dia_semana(x_log)

        # Ajuste linear por SKU: y_resid ~ beta * x_resid
        y_arr = y_resid.to_numpy()
        x_arr = x_resid.to_numpy()
        n_cols = y_arr.shape[1]
        beta = np.zeros(n_cols)

        for j in range(n_cols):
            xj = x_arr[:, j]
            yj = y_arr[:, j]
            mask = np.isfinite(xj) & np.isfinite(yj)
            if mask.sum() < 30:
                beta[j] = 0.0
                continue
            x_use = xj[mask]
            y_use = yj[mask]
            var_x = float(np.var(x_use))
            if var_x < 1e-10:
                beta[j] = 0.0
                continue
            cov_xy = float(np.mean((x_use - x_use.mean()) * (y_use - y_use.mean())))
            beta[j] = cov_xy / var_x

        y_adj = y_arr - (x_arr * beta)
        return pd.DataFrame(y_adj, index=idx, columns=cols)

    def detectar_arestas(
        self,
        min_venda_total_anchor: float = 100000,
        min_dias_stockout_anchor: int = 12,
        min_effect_d: float = 0.6,
        min_uplift: float = 0.3,
        min_media_base_candidato: float = 5.0,
    ) -> pd.DataFrame:
        sku_stats = (
            self.df.groupby("sku")
            .agg(venda_total=("quantidade", "sum"), dias_stockout=("stockout", "sum"))
            .reset_index()
        )

        anchors = sku_stats[
            (sku_stats["venda_total"] >= min_venda_total_anchor)
            & (sku_stats["dias_stockout"] >= min_dias_stockout_anchor)
        ]["sku"].tolist()

        edges = []
        for a in anchors:
            out_days = self.stock.index[self.stock[a]]
            in_days = self.stock.index[~self.stock[a]]

            if len(out_days) < min_dias_stockout_anchor:
                continue

            x_out = self.signal.loc[out_days]
            x_in = self.signal.loc[in_days]

            diff = x_out.mean(axis=0) - x_in.mean(axis=0)
            pooled = np.sqrt((x_out.var(axis=0, ddof=1) + x_in.var(axis=0, ddof=1)) / 2)
            effect_d = diff / pooled.replace(0, np.nan)
            t_result = stats.ttest_ind(
                x_out.to_numpy(),
                x_in.to_numpy(),
                axis=0,
                equal_var=False,
                nan_policy="omit",
            )
            p_two_sided = pd.Series(t_result.pvalue, index=self.sales.columns)
            t_stat = pd.Series(t_result.statistic, index=self.sales.columns)
            p_one_sided = pd.Series(
                np.where(t_stat > 0, p_two_sided / 2.0, 1.0 - (p_two_sided / 2.0)),
                index=self.sales.columns,
            )

            m_out = self.sales.loc[out_days].mean(axis=0)
            m_in = self.sales.loc[in_days].mean(axis=0)
            uplift = (m_out - m_in) / m_in.replace(0, np.nan)

            tbl = pd.DataFrame(
                {
                    "sku_anchor": a,
                    "sku_candidato": self.sales.columns,
                    "effect_d": effect_d.values,
                    "uplift_pct": uplift.values,
                    "media_base_candidato": m_in.values,
                    "dias_stockout_anchor": len(out_days),
                    "p_value": p_one_sided.values,
                }
            )
            tbl = tbl[tbl["sku_candidato"] != a]
            tbl = tbl[
                (tbl["effect_d"] >= min_effect_d)
                & (tbl["uplift_pct"] >= min_uplift)
                & (tbl["media_base_candidato"] >= min_media_base_candidato)
            ]
            edges.append(tbl)

        self.edges = (
            pd.concat(edges, ignore_index=True)
            if edges
            else pd.DataFrame(
                columns=[
                    "sku_anchor",
                    "sku_candidato",
                    "effect_d",
                    "uplift_pct",
                    "media_base_candidato",
                    "dias_stockout_anchor",
                    "p_value",
                    "p_value_ajustado",
                ]
            )
        )
        if not self.edges.empty:
            self.edges["p_value_ajustado"] = self._ajustar_fdr_bh(self.edges["p_value"])
        return self.edges

    def detectar_pares_bidirecionais(self) -> pd.DataFrame:
        if self.edges.empty:
            self.pares = pd.DataFrame(
                columns=["sku1", "sku2", "effect_12", "effect_21", "uplift_12", "uplift_21", "score"]
            )
            return self.pares

        ab = self.edges.rename(
            columns={
                "sku_anchor": "sku1",
                "sku_candidato": "sku2",
                "effect_d": "effect_12",
                "uplift_pct": "uplift_12",
            }
        )
        ba = self.edges.rename(
            columns={
                "sku_anchor": "sku2",
                "sku_candidato": "sku1",
                "effect_d": "effect_21",
                "uplift_pct": "uplift_21",
            }
        )

        pares = ab.merge(ba, on=["sku1", "sku2"], how="inner")
        pares = pares[pares["sku1"] < pares["sku2"]].copy()
        pares["score"] = np.minimum(pares["effect_12"], pares["effect_21"]) + 0.3 * np.minimum(
            pares["uplift_12"], pares["uplift_21"]
        )
        pares = pares.sort_values("score", ascending=False)

        self.pares = pares[["sku1", "sku2", "effect_12", "effect_21", "uplift_12", "uplift_21", "score"]]
        return self.pares

    def construir_grupos(self, min_score: float = 1.0, min_tamanho_grupo: int = 3) -> pd.DataFrame:
        if self.pares.empty:
            self.grupos = pd.DataFrame(columns=["grupo_id", "sku", "tamanho_grupo"])
            return self.grupos

        fortes = self.pares[self.pares["score"] >= min_score][["sku1", "sku2"]]
        if fortes.empty:
            self.grupos = pd.DataFrame(columns=["grupo_id", "sku", "tamanho_grupo"])
            return self.grupos

        parent = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for r in fortes.itertuples(index=False):
            union(r.sku1, r.sku2)

        grupos = {}
        for sku in set(fortes["sku1"]).union(set(fortes["sku2"])):
            root = find(sku)
            grupos.setdefault(root, []).append(sku)

        linhas = []
        gid = 1
        for skus in sorted(grupos.values(), key=len, reverse=True):
            if len(skus) < min_tamanho_grupo:
                continue
            for s in sorted(skus):
                linhas.append({"grupo_id": gid, "sku": s, "tamanho_grupo": len(skus)})
            gid += 1

        self.grupos = pd.DataFrame(linhas)
        return self.grupos

    def montar_saida_direcional(self) -> pd.DataFrame:
        if self.edges.empty:
            return self.edges.copy()

        saida = self.edges.copy()
        saida["par_id"] = saida.apply(
            lambda r: "|".join(sorted([str(r["sku_anchor"]), str(r["sku_candidato"])])), axis=1
        )
        saida["tem_reciproca"] = False
        saida["effect_reciproco"] = np.nan
        saida["uplift_reciproco"] = np.nan
        saida["score_par"] = np.nan

        if self.pares.empty:
            return saida

        pares_idx = self.pares.set_index(["sku1", "sku2"])
        reciproc_map = {}
        for (sku1, sku2), row in pares_idx.iterrows():
            reciproc_map[(sku1, sku2)] = {
                "effect_reciproco": row["effect_21"],
                "uplift_reciproco": row["uplift_21"],
                "score_par": row["score"],
            }
            reciproc_map[(sku2, sku1)] = {
                "effect_reciproco": row["effect_12"],
                "uplift_reciproco": row["uplift_12"],
                "score_par": row["score"],
            }

        for i, r in saida[["sku_anchor", "sku_candidato"]].iterrows():
            chave = (r["sku_anchor"], r["sku_candidato"])
            if chave in reciproc_map:
                saida.at[i, "tem_reciproca"] = True
                saida.at[i, "effect_reciproco"] = reciproc_map[chave]["effect_reciproco"]
                saida.at[i, "uplift_reciproco"] = reciproc_map[chave]["uplift_reciproco"]
                saida.at[i, "score_par"] = reciproc_map[chave]["score_par"]

        return saida

    def exportar_resultados(self, pasta_saida: str = "saida_substitutos") -> None:
        out = Path(pasta_saida)
        out.mkdir(parents=True, exist_ok=True)

        direcional = self.montar_saida_direcional()
        direcional.to_csv(out / "substituicao_direcional.csv", index=False, encoding="utf-8-sig")

        print(f"Arestas: {len(self.edges)}")
        print(f"Pares bidirecionais: {len(self.pares)}")
        print(f"Arestas com reciproca: {int(direcional['tem_reciproca'].sum()) if not direcional.empty else 0}")
        print(f"Arquivo salvo em: {(out / 'substituicao_direcional.csv').resolve()}")


if __name__ == "__main__":
    modelo = ModeloSubstitutos("produtos_substitutos_base.csv")
    modelo.detectar_arestas()
    modelo.detectar_pares_bidirecionais()
    modelo.exportar_resultados()
