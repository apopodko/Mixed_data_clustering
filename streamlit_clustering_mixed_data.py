import pandas as pd
import numpy as np
import time
import random
import streamlit as st

# plots
import seaborn as sns
import matplotlib.pyplot as plt

# HAC
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

# HDBSCAN
import hdbscan
from hdbscan.validity import validity_index

# FeatureEncode
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
#from category_encoders import CatBoostEncoder

# Feature weights
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Cluster Explanation
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import faiss

# Расчет ренжей для расстояния Говера
def compute_numeric_ranges(df, num_cols, method='minmax', iqr_multiplier=1.5):
    """
    Возвращает словарь {col: range} для нормализации при расчете расстояния Говера
      - 'minmax'  : range = max - min
      - 'iqr' : range = (Q3-Q1) * iqr_multiplier
    """
    ranges = {}
    for c in num_cols:
        col = df[c].dropna().to_numpy(dtype=float)
        if col.size == 0:
            ranges[c] = 1.0
            continue

        if method == 'minmax':
            r = float(np.max(col) - np.min(col))
        elif method == 'iqr':
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            r = float((q3 - q1) * iqr_multiplier)
        else:
            raise ValueError("unknown method")

        # Если range слишком мал, то ставим его 1.0
        if not np.isfinite(r) or r <= 1e-8:
            r = 1.0
        ranges[c] = r
    return ranges

# Расчет весов для расстояния Говера
def compute_feature_weights(df, target, num_cols, cat_cols, task="regression"):
    """
    Считает веса признаков по взаимной информации (MI) с таргетом
    df : Входные данные
    target : pd.Series или np.array таргет (числовой или категориальный)
    task : str
        "regression" или "classification".

    output:
    weights : dict
        {column_name: weight}, веса нормированы (сумма = число признаков).
    """
    df = df.copy()
    y = target.values if isinstance(target, pd.Series) else np.array(target)

    # Кодируем категориальные
    df_enc = df.copy()
    discrete_features = []
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        discrete_features.append(df.columns.get_loc(c))  # индексы категориальных

    X = df_enc.to_numpy()

    # Вычисляем MI
    if task == "regression":
        mi = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
    elif task == "classification":
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
    else:
        raise ValueError("task должен быть 'regression' или 'classification'")

    # Нормируем веса (сумма = число признаков)
    mi = np.maximum(mi, 1e-9)  # чтобы не было нулевых
    mi_normalized = mi / mi.sum() * len(mi)

    weights = {col: w for col, w in zip(df.columns, mi_normalized)}
    return weights

# Расчет расстояния Говера
def compute_gower_matrix(df, num_cols, cat_cols, numeric_ranges=None,
                          weights=None, dtype=np.float32):
    """
    - numeric_ranges: dict col->range. Если None, computed by minmax on df.
    - weights: dict col->weight. Если None - все 1
    """
    X = df.reset_index(drop=True)
    n = len(X)
    if n == 0:
        return np.zeros((0,0), dtype=dtype)

    if numeric_ranges is None:
        numeric_ranges = compute_numeric_ranges(X, num_cols, method='minmax')

    D = np.zeros((n, n), dtype=np.float64)  # accumulate in float64 for numeric stability

    # weights
    if weights is None:
        # equal weight per feature
        w_num = {c: 1.0 for c in num_cols}
        w_cat = {c: 1.0 for c in cat_cols}
    else:
        # user-provided; missing keys default to 1.0
        w_num = {c: float(weights.get(c, 1.0)) for c in num_cols}
        w_cat = {c: float(weights.get(c, 1.0)) for c in cat_cols}

    # numeric part
    for c in num_cols:
        col = X[c].to_numpy(dtype=float)
        rng = numeric_ranges.get(c, 1.0)
        # normalized differences (broadcast)
        mat = np.abs(col[:, None] - col[None, :]) / rng
        D += w_num.get(c, 1.0) * mat

    # categorical part
    for c in cat_cols:
        col = X[c].astype(str).to_numpy()
        neq = (col[:, None] != col[None, :]).astype(np.float64)
        D += w_cat.get(c, 1.0) * neq

    # normalize by total weights sum
    total_weight = sum(w_num.values()) + sum(w_cat.values())
    if total_weight <= 0:
        total_weight = 1.0
    D = (D / float(total_weight)).astype(dtype)
    return D

### Кластеризация
def tune_hdbscan(D, param_grid, d):
    best_score = -1
    best_params = None
    best_labels = None

    for params in param_grid:
        cl = hdbscan.HDBSCAN(metric="precomputed", **params)
        labels = cl.fit_predict(D)

        if len(set(labels)) > 1:
            score = validity_index(
                D,
                labels,
                metric="precomputed",
                d=d
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels
    return best_params, best_score, best_labels

def clusterize(df, D, method="hac", max_k=20):
    """
    Кластеризация на подвыборке.
    method = "hdbscan" или "hac"
    """

    if method == "hdbscan":
        cl = hdbscan.HDBSCAN(
            metric="precomputed"
        )
        # specify parameters and distributions to sample from
        param_grid = [
            {"min_cluster_size": mcs, "min_samples": ms, "cluster_selection_method": csm}
            for mcs in [10, 20, 30, 50, 100]
            for ms in [5, 10, 20, 30, 50, 100]
            for csm in ["eom", "leaf"]
        ]
        d = df.shape[1]
        best_params, best_score, labels = tune_hdbscan(
            D.astype(np.float64), param_grid, d)

        print("Best params:", best_params)
        print("Best validity:", best_score)

    elif method == "hac":
        # преобразуем матрицу в condensed form
        condensed = squareform(D, checks=False)

        # linkage (average linkage лучше для Gower)
        Z = linkage(condensed, method="complete")

        # поиск оптимального k по silhouette
        best_k, best_score, best_labels = None, -1, None
        for k in range(2, min(max_k, len(df)//100)):
            labels = fcluster(Z, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = silhouette_score(D, labels, metric="precomputed")
            except Exception:
                continue
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels

        if best_labels is None:
            # fallback: всё в один кластер
            best_labels = np.ones(len(df), dtype=int)

        labels = best_labels

    else:
        raise ValueError("method must be 'hdbscan' or 'hac'")

    return labels

### Векторизация данных для FAISS
def vectorize(df, num_cols, cat_cols, ohe_thresh=20, hash_dim=32):
    """
    Преобразует таблицу в числовой массив для кластеризации / поиска соседей.

    Parameters:
        df : pd.DataFrame
        numeric_cols : list[str] - числовые колонки
        cat_cols : list[str] - категориальные колонки
        ohe_thresh : int - макс число уникальных категорий для OHE
        hash_dim : int - размер выходного вектора для FeatureHasher

    output:
        X : np.ndarray, float32
    """
    # --- числовые ---
    scaler = RobustScaler()
    X_num = scaler.fit_transform(df[num_cols]) if num_cols else None

    # --- категориальные ---
    X_cat_list = []
    encoders = {}

    for c in cat_cols:
        n_unique = df[c].nunique()
        col_data = df[[c]].astype(str)

        if n_unique <= ohe_thresh:
            # OHE
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_c = enc.fit_transform(col_data)
            encoders[c] = enc
        else:
            # FeatureHasher
            enc = FeatureHasher(n_features=hash_dim, input_type='string')
            # FeatureHasher принимает список словарей: [{cat_val: 1}, ...]
            X_c = enc.transform([{val: 1} for val in col_data[c]]).toarray()
            encoders[c] = enc
        X_cat_list.append(X_c)

    if X_cat_list:
        X_cat = np.hstack(X_cat_list)
    else:
        X_cat = None

    # --- объединение ---
    if X_num is not None and X_cat is not None:
        X = np.hstack([X_num, X_cat])
    elif X_num is not None:
        X = X_num
    else:
        X = X_cat

    return X.astype(np.float32)

######################## Streamlit layout ##################

st.set_page_config(page_title="Анализ и фильтрация данных", layout="wide")

st.title("Анализ и фильтрация данных")

# === 1. Загрузка файла ===
uploaded_file = st.file_uploader("Загрузите CSV или Excel файл", type=["csv", "xlsx"])
with st.expander("Если есть проблемы с форматом признаков или разделителем"):
  option_decimal = st.radio(
      "Выберите десятичный разделитель",
      (".", ","),
      horizontal=True)
  option_sep = st.text_input("Введите свой тип разделителя для CSV", value=",")

if uploaded_file is not None:
    # Определяем формат и читаем файл

    if uploaded_file.name.endswith(".csv"):
        if option_decimal == ".":
            option_thousand = ','
        else:
            option_thousand = None
        try:
          df = pd.read_csv(
              uploaded_file,
              sep=option_sep,
              decimal=option_decimal,
              thousands=option_thousand)
        except (ValueError, TypeError):
          st.error('Проблема в чтении файла', icon="🚨")
    else:
        df = pd.read_excel(uploaded_file)

    # === 2. Преобразуем даты автоматически ===
    for col in df.select_dtypes(exclude=[np.number]).columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            pass  # не дата — пропускаем

    st.subheader("Предпросмотр данных и их типы")
    st.dataframe(df.head())
    st.dataframe(df.dtypes.to_frame().T)

    # Подготовка данных
    st.subheader("Подготовка данных")

    # === 3. Выбор таргета ===
    target = st.selectbox("Выберите целевой признак", df.columns.tolist())
    df.dropna(subset=[target], inplace=True)
    option_task = st.radio("Выберите тип целевого признака",
                           ("Числовой", "Категориальный"),
                           horizontal=True)
    if option_task == "Числовой":
      option_task = "regression"
    else:
      option_task = "classification"

    # === 4. Выбор столбцов для удаления ===
    drop_cols = st.multiselect("Выберите признаки для удаления",
                               df.drop(target, axis=1).columns.tolist())
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # === 5. Фильтрация ===
    filter_cols = st.multiselect("Выберите признаки для фильтра", df.columns.tolist())
    if filter_cols:

      with st.expander("Открыть фильтры"):
          for col in filter_cols:
              col_type = df[col].dtype

              if pd.api.types.is_datetime64_any_dtype(col_type):
                  # фильтр по дате
                  min_date, max_date = df[col].min(), df[col].max()
                  start_date, end_date = st.date_input(
                      f"{col}: диапазон дат",
                      [min_date, max_date],
                      key=col
                  )
                  mask = (df[col] >= pd.to_datetime(start_date)) & (df[col] <= pd.to_datetime(end_date))
                  df = df.loc[mask]

              elif pd.api.types.is_numeric_dtype(col_type):
                  # фильтр по числовому диапазону
                  min_val, max_val = float(df[col].min()), float(df[col].max())
                  val_range = st.slider(
                      f"{col}: диапазон значений",
                      min_val, max_val, (min_val, max_val),
                      key=col
                  )
                  mask = (df[col] >= val_range[0]) & (df[col] <= val_range[1])
                  df = df.loc[mask]

              else:
                  # фильтр по категориям/тексту
                  unique_vals = df[col].dropna().unique().tolist()
                  if len(unique_vals) > 1 and len(unique_vals) <= 100:  # ограничим слишком большие списки
                      selected_vals = st.multiselect(
                          f"{col}: выберите значения",
                          unique_vals,
                          default=unique_vals,
                          key=col
                      )
                      df = df[df[col].isin(selected_vals)]

    # === 6. Работа с пропусками ===
    missing_option = st.radio("Что делать с пропусками в данных?",
      ("Заполнить медианой и MISSING", "Выбросить объекты с пропусками"),
      horizontal=True)
    num_cols = (df
                .drop(target, axis=1)
                .select_dtypes(include=[np.number])
                .columns.tolist())
    cat_cols = (df
                .drop(target, axis=1)
                .select_dtypes(exclude=[np.number, np.datetime64])
                .columns.tolist())

    if missing_option == "Заполнить медианой и MISSING":
      # Заполняем пропуски медианой, либо __MISSING__
      for col in num_cols:
          df[col] = df[col].fillna(df[col].median())
      for col in cat_cols:
          df[col] = df[col].fillna("__MISSING__").astype(str)
    else:
      # Выбрасываем все объекты с пропусками
      df = df.dropna()

    # === 7. Результат ===
    st.subheader("Отфильтрованные данные")
    st.write(f"Отображается {len(df)} из {len(df)} строк")
    st.dataframe(df)

    # Сохраняем данные для кластеризации
    X = df.drop(target, axis=1)
    y = df[target]


    ############# Кластеризация ################################
    st.header("Кластеризация")
    ### Считаем расстояние Говера
    st.subheader("Рассчет расстояния Говера")

    option_weights = st.radio("Рассчитываем веса на основе взаимной ифнормации с целевым признаком?",
                              ("Да", "Нет"),
                              horizontal=True)
    
    if st.button("Рассчитать расстояния Говера"):


      # Векторизуем данные для FAISS  
      X_vectorized = vectorize(X, num_cols, cat_cols)
      N = X.shape[0]
      S = min(8000, N)
      idx_all = np.arange(N)
      idx_S = np.random.choice(idx_all, size=S, replace=False)
      idx_rest = np.setdiff1d(idx_all, idx_S)

      X_S = X_vectorized[idx_S]
      X_rest = X_vectorized[idx_rest]

      if option_weights == "Да":
        weights = compute_feature_weights(X.iloc[idx_S], y.iloc[idx_S], num_cols, cat_cols, task=option_task)
      else:
        weights = None       
      st.session_state.idx_S = idx_S
      st.session_state.D = compute_gower_matrix(X.iloc[idx_S], num_cols, cat_cols, numeric_ranges=None, weights=weights)
      st.text(weights)

    #### Калстеризуем
    st.subheader("Кластеризация на основе расстояния Говера")
    option_method = st.radio("Метод кластеризации",
                              ("HDBSCAN", "HAC"),
                              horizontal=True)
    
    if st.button("Начать кластеризацию"):

      if option_method=="HDBSCAN":
        option_method=="hdbscan"
      else:
        option_method=="hac"
      labels = clusterize(X.iloc[st.session_state.idx_S], st.session_state.D, method="hac", max_k=20)
      st.text(labels)
else:
    st.info("Загрузите файл для начала работы")