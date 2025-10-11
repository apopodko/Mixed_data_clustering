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

from tempfile import NamedTemporaryFile
from supertree import SuperTree
from streamlit.components.v1 import html
import faiss

# Расчет ренжей для расстояния Говера
def compute_numeric_ranges(df, num_cols, method='minmax'):
    """
    Возвращает словарь {col: range} для нормализации при расчете расстояния Говера
      - 'minmax'  : range = max - min
      - 'iqr' : range = (Q3-Q1)
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
            r = float((q3 - q1))
        else:
            raise ValueError("unknown method")

        # Если range слишком мал, то ставим его 1.0
        if not np.isfinite(r) or r <= 1e-8:
            r = 1.0
        ranges[c] = r
    return ranges

# Расчет весов для расстояния Говера
def compute_feature_weights(df, target, num_cols, cat_cols):
    """
    Считает веса признаков по взаимной информации (MI) с таргетом
    df : Входные данные
    target : pd.Series или np.array таргет (числовой или категориальный)

    output:
    weights : dict
        {column_name: weight}, веса нормированы (сумма = число признаков).
    """
    y = target.values

    # Кодируем категориальные
    df_enc = df.copy()
    discrete_features = []
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        discrete_features.append(df.columns.get_loc(c))  # индексы категориальных
    X = df_enc.to_numpy()

    # Вычисляем MI
    if pd.api.types.is_numeric_dtype(target) == True:
        mi = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
    elif pd.api.types.is_object_dtype(target) == True:
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
    else:
        raise ValueError("Тип целевой переменной должен быть числовым или категориальным")

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
    return D.astype(np.float64)

### Кластеризация
def clusterize(df, D, method="HDBSCAN", max_k=20):
    """
    Кластеризация на подвыборке.
    method = "hdbscan" или "hac"
    """

    if method == "HDBSCAN":
      best_score = -1
      best_params = None
      best_labels = None
      d = df.shape[1]
      # specify parameters and distributions to sample from
      param_grid = [
          {"min_cluster_size": mcs, "min_samples": ms}
          for mcs in [10, 20, 30, 50]
          for ms in [10, 20, 30, 50]
      ]

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

    elif method == "HAC":
      # преобразуем матрицу в condensed form
      condensed = squareform(D, checks=False)

      # linkage (average linkage лучше для Gower)
      Z = linkage(condensed, method="complete")

      # поиск оптимального k по silhouette
      best_k, best_score, best_labels = None, -1, None
      for k in range(2, min(max_k, len(df)//10)):
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
      raise ValueError("method must be 'HDBSCAN' or 'HAC'")

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

#### FAISS на сэпмлированной выборке
def build_faiss_hnsw(X_S, M=32, efSearch=64):
    """
    Создаёт HNSW индекс FAISS для подвыборки X_S

        X_S : сэмплированный массив от кластерзиации
        M : число соседей в графе HNSW
        efSearch : ширина поиска
    output:
        index : faiss.IndexHNSWFlat
    """
    dim = X_S.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efSearch = efSearch
    index.add(X_S)
    return index


###### Присваивание лейблом через FAISS HSNW
def assign_faiss(index, X_rest, labels_S, k=3, ood_threshold=None, weighted=True):
    """
    Приписывает новые точки к кластерам из labels_S через HNSW

        index : faiss.IndexHNSWFlat, построенный на кластеризованном сэмпле X_S
        X_rest : новые точки из некластеризованной выборки
        labels_S : метки кластеров
        k : число ближайших соседей
        ood_threshold : float или None, порог для OOD; если None - вычисляется как 99-й перцентиль расстояний в S
        weighted : bool, использовать взвешенное голосование по расстоянию

    output:
        assigned_labels : присвоенные лейбл
        assign_distance : расстояние до выбранного кластера
        is_OOD : True если объект OOD
    """
    n_S = labels_S.shape[0]
    k = min(k, n_S)

    D, I = index.search(X_rest, k)  # distances & indices
    n_rest = X_rest.shape[0]
    assigned_labels = np.empty(n_rest, dtype=int)
    assign_distance = np.empty(n_rest, dtype=float)
    is_OOD = np.zeros(n_rest, dtype=bool)

    # OOD threshold если не задан
    if ood_threshold is None:
        # эмпирический 99-й перцентиль всех расстояний внутри S
        # D_self: расстояния до 1-го ближайшего соседа в S (exclude self)
        D_self, _ = index.search(index.reconstruct_n(0, n_S), 2)
        D_self = D_self[:, 1]  # первый сосед = сам объект, берем 2-й
        ood_threshold = np.quantile(D_self, 0.99)

    for i in range(n_rest):
        neigh_idxs = I[i]
        neigh_dists = D[i]
        neigh_labels = labels_S[neigh_idxs]

        if weighted:
            # взвешенное голосование: 1/(dist+1e-9)
            weights = 1.0 / (neigh_dists + 1e-9)
            label_score = {}
            for lbl, w in zip(neigh_labels, weights):
                label_score[lbl] = label_score.get(lbl, 0.0) + w
            # выбираем label с максимальной суммы весов
            assigned_label = max(label_score.items(), key=lambda x: x[1])[0]
        else:
            # через ArgMax
            vals, counts = np.unique(neigh_labels, return_counts=True)
            assigned_label = vals[np.argmax(counts)]

        assigned_labels[i] = assigned_label
        # расстояние до ближайшего соседа с выбранным label
        mask = neigh_labels == assigned_label
        assign_distance[i] = neigh_dists[mask].min()
        # OOD
        is_OOD[i] = assign_distance[i] > ood_threshold

    return assigned_labels, assign_distance, is_OOD

#################### Анализ кластеризации ##################################

### Общий анализ кластеров
def analyze_all_clusters(df, target_col, num_cols, cat_cols, cluster_col="cluster", max_depth=6):
    results = {}
    df.reset_index(inplace=True, drop=True)
    # Распределение таргета по кластерам
    if pd.api.types.is_numeric_dtype(df[target_col]):
      summary = df.groupby(cluster_col)[target_col].agg(
          ["count", "mean", "std", "min", "max", "median"]
      ).sort_values("mean", ascending=False)
      summary.columns = [
          "Количество объектов",
          "Среднее значение таргета",
          "Стд. откл.",
          "Минимальное значение",
          "Максимальное",
          "Медиана"
      ]

    else:
      counts = (
          df.groupby([cluster_col, target_col])
          .size()
          .reset_index(name="count")
      )

      total = counts.groupby(cluster_col)["count"].transform("sum")
      counts["share"] = (counts["count"] / total * 100).round(2)

      # Наиболее частая категория в каждом кластере
      mode_df = (
          counts.sort_values(["cluster", "count"], ascending=[True, False])
          .drop_duplicates(subset=[cluster_col])
          .rename(columns={target_col: "most_frequent"})
          .loc[:, [cluster_col, "most_frequent"]]
      )

      summary = (
          counts.pivot(index=cluster_col, columns=target_col, values="share")
          .fillna(0)
      )
      summary.columns = [f"{col} (% в кластере)" for col in summary.columns]
      summary["Самая частая категория"] = mode_df.set_index(cluster_col)["most_frequent"]
      summary["Всего объектов"] = counts.groupby(cluster_col)["count"].sum().values

    # Объяснение кластеров через DecisionTreeClassifier
    X = df[num_cols + cat_cols]
    y = df[cluster_col].astype(str)

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )

    clf = Pipeline([
        ("prep", preproc),
        ("tree", DecisionTreeClassifier(max_depth=max_depth, random_state=42))
    ])

    clf.fit(X, y)
    #### Строим дерево объяснющее кластеры
    super_tree = SuperTree(
        clf.named_steps["tree"],
        clf.named_steps["prep"].transform(X),
        y,
        clf.named_steps["prep"].get_feature_names_out(),
        np.unique(y))


    importances = pd.Series(
        clf.named_steps["tree"].feature_importances_,
        index=clf.named_steps["prep"].get_feature_names_out()
    ).sort_values(ascending=False)
    importances.name = "Важность признака"

    results["cluster_importances"] = importances

    return summary, super_tree, importances

######
### Общий анализ внутри кластеров
def analyze_within_clusters(df, target_col, num_cols, cat_cols, cluster_col="cluster", cluster=0, max_depth=5):
    results = {}
    df.reset_index(inplace=True, drop=True)

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )

    st.header("Внутрикластерный анализ")
    # Анализ таргета внутри каждого кластера
    target_results = {}
    subset = df[df[cluster_col] == cluster]

    X_sub = subset[num_cols + cat_cols]
    y_sub = subset[target_col]

    # Если целевой признак числовой DecisionTreeRegresor
    if pd.api.types.is_numeric_dtype(subset[target_col]) == True:

      model = Pipeline([
          ("prep", preproc),
          ("tree", DecisionTreeRegressor(max_depth=max_depth, random_state=42))
      ])

      model.fit(X_sub, y_sub)
      class_names = None

    # Если целевой признак категориальный DecisionTreeClassifier
    elif pd.api.types.is_object_dtype(subset[target_col]) == True:

      le = LabelEncoder()
      y_enc = le.fit_transform(y_sub)

      model = Pipeline([
          ("prep", preproc),
          ("tree", DecisionTreeClassifier(max_depth=max_depth, random_state=42))
      ])

      model.fit(X_sub, y_enc)
      class_names = le.classes_

    else:
        raise ValueError("Тип целевой переменной должен быть числовым или категориальным")

    importances = pd.Series(
        model.named_steps["tree"].feature_importances_,
        index=model.named_steps["prep"].get_feature_names_out()
    ).sort_values(ascending=False)
    importances.name = "Важность признака"

    target_results[cluster] = importances

    st.subheader(f"\n Кластер {cluster}: факторы и правила, влияющие на таргет")
    st.dataframe(importances.head(5))

    super_tree = SuperTree(
        model.named_steps["tree"],
        model.named_steps["prep"].transform(X_sub),
        y_sub.reset_index(drop=True),
        model.named_steps["prep"].get_feature_names_out(),
        target_names=class_names)

    return super_tree

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
    st.write("Внимательно проверьте типы признаков")
    st.dataframe(df.dtypes.to_frame().T)

    # Подготовка данных
    st.subheader("Подготовка данных")

    # === 3. Выбор столбцов для приведения к числовому типу ===
    to_num_cols = st.multiselect("Выберите признаки для приведения к числовому типу",
                                df.columns.tolist())    
    df[to_num_cols] = df[to_num_cols].apply(pd.to_numeric, errors='coerce')

    # === 4. Выбор столбцов для удаления ===
    drop_cols = st.multiselect("Выберите признаки для удаления, и не забудьте про индексные признаки",
                               df.columns.tolist())
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # === 3. Выбор таргета ===
    target = st.selectbox("Выберите целевой признак", df.columns.tolist())
    if  df[target].nunique() > 1000 and pd.api.types.is_object_dtype(df[target]):
      st.error("Целевой признак с очень высокой кардинальностью, выберите другой признак")
    df.dropna(subset=[target], inplace=True)

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

    st.write("У признаков с кардинальностью больше 100 редкие категории автоматически переназначаются на 'Other'!")
    for col in cat_cols:
      if df[col].nunique() > 100:
        top_cat = df[col].value_counts().head(20).index.tolist()
        df[col] = np.where(df[col].isin(top_cat), df[col], 'Other')

    df = df.reset_index(drop=True)
    st.write(f"Отображается {len(df.head(500))} из {len(df)} строк")
    st.dataframe(df.head(500))
    st.write("Еще раз внимательно проверьте типы признаков")
    st.dataframe(df.dtypes.to_frame().T)

    # Сохраняем данные для кластеризации
    datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
    X = df.drop(target, axis=1)
    X = X.drop(datetime_columns, axis=1)
    y = df[target]

    ############# Кластеризация ################################
    st.header("Кластеризация")
    ### Считаем расстояние Говера
    st.subheader("Рассчет расстояния Говера")

    option_weights = st.radio("Рассчитываем веса на основе взаимной информации с целевым признаком?",
                              ("Нет", "Да"),
                              horizontal=True)

    if st.button("Рассчитать расстояния Говера"):
      # Готовим индексы для FAISS, на случай, несли выборка будет больше 8000
      N = X.shape[0]
      S = min(8000, N)
      idx_all = np.arange(N)
      idx_S = np.random.choice(idx_all, size=S, replace=False)
      idx_rest = np.setdiff1d(idx_all, idx_S)
      st.session_state.idx_S = idx_S
      st.session_state.idx_rest = idx_rest
      # Рассчитываем веса для расстояния Говера
      if option_weights == "Да":
        weights = compute_feature_weights(X.iloc[idx_S], y.iloc[idx_S], num_cols, cat_cols)
      else:
        weights = None

      # Рассчитываем расстояния Говера
      st.session_state.D = compute_gower_matrix(
          X.iloc[idx_S],
          num_cols,
          cat_cols,
          numeric_ranges=None,
          weights=weights
      )
      st.success("Матрица рассчитана")

    #### Кластеризуем
    st.subheader("Кластеризация на основе расстояния Говера")
    option_method = st.radio("Метод кластеризации",
                              ("HDBSCAN", "HAC"),
                              horizontal=True)

    if st.button("Начать кластеризацию"):
      if hasattr(st.session_state, "D"):
        st.session_state.labels = clusterize(
            X.iloc[st.session_state.idx_S],
            st.session_state.D,
            method=option_method
        )
      else:
        st.error("Рассчитайте расстонния Говера")

      # --- FAISS HNSW index если размер выборки > 8000 ---
      if X.shape[0] > 8000:
        st.subheader("FAISS HNSW разметка лейблов")
        X_vectorized = vectorize(X, num_cols, cat_cols)
        X_S = X_vectorized[st.session_state.idx_S]
        X_rest = X_vectorized[st.session_state.idx_rest]
        index = build_faiss_hnsw(X_S, M=32, efSearch=64)

        # --- assign остальных ---
        labels_rest, dist_rest, is_ood = assign_faiss(index,
                                                      X_rest,
                                                      st.session_state.labels,
                                                      k=3
                                                      )

        # --- собрать результат ---
        result = df.copy()
        result['cluster'] = np.nan
        result['assign_distance'] = np.nan
        result['is_OOD'] = False

        # метки для подвыборки
        result.loc[st.session_state.idx_S, 'cluster'] = st.session_state.labels
        result.loc[st.session_state.idx_S, 'assign_distance'] = 0.0
        result.loc[st.session_state.idx_S, 'is_OOD'] = False

        # метки для остальных
        result.loc[st.session_state.idx_rest, 'cluster'] = labels_rest
        result.loc[st.session_state.idx_rest, 'assign_distance'] = dist_rest
        result.loc[st.session_state.idx_rest, 'is_OOD'] = is_ood
        st.session_state.result = result
        st.success(f"Кластеризация методом {option_method} с доразметкой FAISS выполнена успешно")

      else:
        result = df.copy()
        st.write(f"Отображается {len(result.head(500))} из {len(result)} строк")
        st.dataframe(result.head(500))    
        result.loc[st.session_state.idx_S, 'cluster'] = st.session_state.labels
        st.session_state.result = result
        st.success(f"Кластеризация методом {option_method} выполнена успешно")

    ##### Результат кластеризации
    if hasattr(st.session_state, "result"):
      st.subheader("Кластеризованные данные")
      st.write(f"Отображается {len(st.session_state.result.head(500))} из {len(st.session_state.result)} строк")
      st.dataframe(st.session_state.result.head(500))

      st.header("Анализ кластеризации")
      if st.button("Начать анализ"):
        st.session_state.summary, st.session_state.super_tree, st.session_state.importances = (
            analyze_all_clusters(
                st.session_state.result,
                target,
                num_cols,
                cat_cols
                )
        )

        with NamedTemporaryFile(suffix=".html", delete=False) as f:
          st.session_state.super_tree.save_html(f.name)
          st.session_state.super_tree_html = open(f.name, "r", encoding="utf-8").read()

      if hasattr(st.session_state, "super_tree"):
        st.subheader("Правила, объясняющие кластеры")

        st.text(f"Распределение целевого признака {target} по кластерам:")
        st.dataframe(st.session_state.summary)
        st.subheader("\n Главные признаки, формирующие кластеры:")
        st.dataframe(st.session_state.importances.head(10))
        if "super_tree_html" in st.session_state:
          html(st.session_state.super_tree_html, height=650)

      option_cluster = st.selectbox(
          "Выберите кластер для внутрикластерного анализа",
          st.session_state.result['cluster'].unique()
      )
      if st.button("Внутрикластерный анализ"):
        cluster_tree = analyze_within_clusters(
            st.session_state.result,
            target,
            num_cols,
            cat_cols,
            cluster=option_cluster
        )
        with NamedTemporaryFile(suffix=".html", delete=False) as f1:
          cluster_tree.save_html(f1.name)
          html(f1.read(), height=650)

else:
    st.info("Загрузите файл для начала работы")