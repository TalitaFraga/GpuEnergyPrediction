import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "batch_size",
    "image_size",
    "dataset_size",
    "epochs",
    "params",
    "flops",
    "layers",
    "model_size_mb",
    "total_train_flops",
]


def build_features(df):
    df = df.copy()

    # FLOPs de uma imagem x quantidade de imagens processadas no treino.
    df["total_train_flops"] = df["flops"] * df["dataset_size"] * df["epochs"]

    X = df[NUMERIC_FEATURES]

    # Só vira coluna quando há mais de uma GPU nos dados.
    if df["gpu_name"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(df["gpu_name"], prefix="gpu", dtype=float)], axis=1)

    return X


def config_groups(df):
    # Repetições da mesma configuração ficam sempre do mesmo lado (treino ou teste).
    return (
        df["architecture"] + "_" + df["batch_size"].astype(str) + "_" +
        df["image_size"].astype(str) + "_" + df["dataset_size"].astype(str) + "_" +
        df["epochs"].astype(str) + "_" + df["gpu_name"]
    )


def evaluate(models, X, y, groups, title):
    n_splits = min(5, groups.nunique())

    if n_splits < 2:
        print(f"\n{title}: grupos insuficientes para validação.")
        return []

    print(f"\n{title} ({n_splits} folds)")

    rows = []

    for name, model in models.items():
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits))

        row = {
            "validation": title,
            "model": name,
            "mae": mean_absolute_error(y, pred),
            "mape_percent": mean_absolute_percentage_error(y, pred) * 100,
            "r2": r2_score(y, pred),
        }
        rows.append(row)

        print(f"  {name:<18} MAE={row['mae']:.3f}  MAPE={row['mape_percent']:.1f}%  R²={row['r2']:.3f}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="energy_j",
        help="coluna a prever (ex.: energy_j, avg_power_w, train_time_seconds)"
    )
    args = parser.parse_args()

    df = pd.read_csv("results/log_summary.csv")
    df = df[df[args.target].notna()].reset_index(drop=True)

    if len(df) < 10:
        print(
            f"Só há {len(df)} experimentos com '{args.target}'. "
            "Rode mais experimentos (de preferência numa GPU que informe potência, como no Colab)."
        )
        return

    X = build_features(df)
    y = df[args.target]

    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    print(f"Alvo: {args.target} | experimentos: {len(df)} | configurações: {config_groups(df).nunique()}")

    metrics = evaluate(models, X, y, config_groups(df), "Configurações não vistas")

    if df["architecture"].nunique() > 1:
        metrics += evaluate(models, X, y, df["architecture"], "Arquiteturas não vistas")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(f"results/model_metrics_{args.target}.csv", index=False)

    final_model = models["RandomForest"].fit(X, y)
    joblib.dump(
        {"model": final_model, "features": list(X.columns), "target": args.target},
        f"results/model_{args.target}.joblib"
    )

    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nImportância das variáveis (RandomForest):")
    print(importances.round(3).to_string())

    print(f"\nMétricas salvas em: results/model_metrics_{args.target}.csv")
    print(f"Modelo salvo em: results/model_{args.target}.joblib")


if __name__ == "__main__":
    main()
