from pathlib import Path

from google.cloud import bigquery

def exportar_csv(output_path: str = "produtos_substitutos_base.csv") -> Path:
    client = bigquery.Client(project="petlove-datausers-prod-01")
    QUERY = """
        WITH estoque AS (
        SELECT
            chv_data_estoque AS data_estoque,
            sku,
            SUM(qt_disponivel_venda_site) AS qt_disponivel
        FROM `petlove-dataeng-prod-01.dw_corporativo.ft_estoque_diario` fed
        JOIN `petlove-dataeng-prod-01.dw_corporativo.dim_produto` p
            ON p.chv_produto = fed.chv_produto
        WHERE chv_data_estoque BETWEEN '2024-01-01' AND CURRENT_DATE - 1
            AND p.erp_subfamilia = 'Ração Úmida'
        GROUP BY 1, 2
        ),
        preco_lista as (
        SELECT 
            date(timestamp_conclusao_pedido) data_pedido,
            sku,
            max(preco.list_price) list_price
        FROM `petlove-dataeng-prod-01.op_pricing.dts_descontos_preco` preco
        WHERE 1=1
            AND date(timestamp_conclusao_pedido) between '2024-01-01' and current_date - 1
        group by 1,2
        ),
        vendas AS (
        SELECT
            dmc.data_pedido,
            dmc.sku,
            SUM(quantidade) AS quantidade,
            sum(receita_bruta_produto) / SUM(quantidade * preco_lista.list_price) - 1 desconto,
            sum(receita_bruta_produto) / SUM(quantidade) preco_venda
        FROM `petlove-dataeng-prod-01.dw_corporativo.dts_margem_conciliacao` dmc
        left join preco_lista on preco_lista.data_pedido = dmc.data_pedido and preco_lista.sku = dmc.sku
        WHERE dmc.data_pedido BETWEEN '2024-01-01' AND CURRENT_DATE - 1
            AND dmc.subfamilia = 'Ração Úmida'
        GROUP BY 1, 2
        )
        SELECT
        vendas.data_pedido,
        vendas.sku,
        vendas.quantidade,
        estoque.qt_disponivel,
        desconto,
        preco_venda
        FROM vendas
        LEFT JOIN estoque ON estoque.data_estoque = vendas.data_pedido AND estoque.sku = vendas.sku
        UNION ALL
        SELECT
        estoque.data_estoque AS data_pedido,
        estoque.sku,
        0 AS quantidade,
        estoque.qt_disponivel,
        null desconto,
        null preco_venda
        FROM estoque
        LEFT JOIN vendas ON estoque.data_estoque = vendas.data_pedido AND estoque.sku = vendas.sku
        WHERE vendas.data_pedido IS NULL
        """
    df = client.query(QUERY).to_dataframe()

    caminho_saida = Path(output_path).resolve()
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    print(f"CSV salvo em: {caminho_saida}")
    print(f"Linhas exportadas: {len(df)}")
    return caminho_saida


if __name__ == "__main__":
    exportar_csv()
