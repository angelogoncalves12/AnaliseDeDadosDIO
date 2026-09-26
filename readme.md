# Análise de detecção de fraudes

## Resumo

O conjunto tem 284.807 transações, mas apenas 0,17% são fraudes. Um modelo que
sempre respondesse "não é fraude" teria quase 100% de acurácia e ainda assim não
encontraria nenhuma fraude. Por isso, também foram avaliadas **precisão**
(quantos alertas eram fraude), **recall** (quantas fraudes foram encontradas) e
**F1-score** (equilíbrio entre os dois).

## Preparação e modelos

Foi mantida a exploração e a Regressão Logística do código-base. Os dados foram
divididos em treino e teste de forma estratificada, para manter a proporção de
fraudes nos dois grupos. Também foi criada a coluna `amount_log`, usando
`log1p(Amount)`, e as colunas `Amount` e `Time` foram padronizadas com
`StandardScaler`.

O primeiro modelo foi uma Regressão Logística, usada como baseline. Depois foram
testados Random Forest, com `class_weight="balanced"`, e XGBoost, com
`scale_pos_weight`. Esses pesos dão mais importância aos poucos exemplos de
fraude, sem criar dados sintéticos.

O `Pipeline` foi usado no XGBoost para apresentar essa estrutura de forma simples.
Não foram usados SMOTE nem busca automática de hiperparâmetros nesta versão, para
manter o tutorial mais fácil de acompanhar e depurar.

## Resultados

Métricas da classe fraude no conjunto de teste:

| Modelo | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: |
| Regressão Logística | 0,86 | 0,68 | 0,76 |
| XGBoost | 0,23 | 0,86 | 0,36 |
| Random Forest | Pendente | Pendente | Pendente |

A Regressão Logística obteve ROC-AUC de `0.9308`. O XGBoost obteve ROC-AUC de
`0.9720` e encontrou mais fraudes, aumentando o recall de 0,68 para 0,86. Em
compensação, sua precisão caiu para 0,23, ou seja, ele gerou mais alertas em
transações que não eram fraude.

Na Regressão Logística foi usado o limiar padrão de `0.50`. No XGBoost foi usado
o threshold `0.30`:

```python
y_pred_xgb = (y_probs_xgb >= 0.30).astype(int)
```

Com esse valor, o modelo sinaliza fraude a partir de 30% de probabilidade. Um
limiar menor tende a aumentar o recall, mas pode reduzir a precisão por aumentar
os falsos positivos.

## SHAP e diferenças em relação ao código-base

A importância nativa da Random Forest indicou como principais variáveis `V14`,
`V12`, `V4`, `V10`, `V17`, `V11`, `V16`, `V7`, `V3` e `V2`. O SHAP foi preparado
com `TreeExplainer` e uma amostra de 100 transações para explicar o impacto das
variáveis em cada previsão.

A execução foi interrompida antes de concluir o gráfico SHAP. Portanto, o ranking
acima é da importância da Random Forest, e não uma conclusão do SHAP. O SHAP ajuda
a explicar previsões, mas não demonstra causalidade.