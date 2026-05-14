import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_preprocess_data():
    # Load files to create all the datasets
    data_transcriptomic = pd.read_pickle('datos_transcriptomicos.pkl')
    data_clinic = pd.read_csv('datos_clinicos_paper.csv')
    data_families = pd.read_csv('Genes_EC_families.csv')

    data_include = data_clinic[['age', 'gender', 'height (cm)', 'ID_Muestra']]
    data_include.set_index('ID_Muestra', inplace=True)

    data_clinic.drop(columns=['tissue', 'statin treatment', 'statin', 'matched cohort statin treatment', 'diabetes treatment', 
                            'diastolic blood pressure (mmhg)', 'systolic blood pressure (mmhg)', 'bmi', 'cholest total (mmol/l)', 
                            'fasting blood glucose (mmol/l)', 'age', 'gender', 'height (cm)', 'Unnamed: 0', 'fasting c-peptide (ngml)'], 
                            inplace=True)

    data_clinic.set_index('ID_Muestra', inplace=True)

    data_transcriptomic = data_transcriptomic.join(data_include)

    families = sorted(data_families['Grupo_Metabolico'].unique())

    data_genes = {}

    for family in families:
        family_genes = data_families[data_families['Grupo_Metabolico'] == family]

        data_genes[family] = data_transcriptomic.iloc[:, family_genes['original_index'].values]

        data_genes[family] = data_genes[family].join(data_include)

    return data_transcriptomic, data_clinic, data_genes

def train_test_validation_split(X, y, architecture_type, validation_size):
    test_size = validation_size
    train_size = 1 - validation_size - test_size
    val_final_ratio = (train_size) / (1 - test_size) 

    # --- 1. PROCESAMIENTO DE ETIQUETAS (Y) ---
    y_proc = y.copy()
    
    # Binarización (Clasificación)
    y_proc['diabetic'] = y_proc['diabetic'].map({'Diabetic': 1, 'Non-Diabetic': 0})
    y_proc['hypertensive'] = y_proc['hypertensive'].map({'Hypertensive': 1, 'Non-Hypertensive': 0})
    
    # Identificar las ÚNICAS columnas de regresión para escalar
    reg_cols = ['hba1c (%)', 'fasting insulin (mui/l)']
    
    # Filtrar solo las que existan
    reg_cols = [c for c in reg_cols if c in y_proc.columns]
    
    # Renombrar columnas
    rename_mapping = {
        'hba1c (%)': 'hba1c',
        'fasting insulin (mui/l)': 'fasting_insulin'
    }
    y_proc.rename(columns=rename_mapping, inplace=True)
    reg_cols = [rename_mapping.get(c, c) for c in reg_cols]

    # Eliminar columnas sobrantes del dataset original para que no estorben
    cols_a_mantener = ['diabetic', 'hypertensive', 'hba1c', 'fasting_insulin']
    y_proc = y_proc[cols_a_mantener]

    def get_target_dict(y_df):
        result = {}
        for col in y_df.columns:
            values = pd.to_numeric(y_df[col], errors='coerce').values.astype('float32')
            result[col] = values
        return result

    # --- LÓGICA AUTOENCODER ---
    if architecture_type == 'Autoencoder':
        X_proc = X.copy()
        X_proc['gender'] = X_proc['gender'].map({'Male': 1, 'Female': 0})

        X_train_val, X_test, y_train_val, y_test = train_test_split(X_proc, y_proc, test_size=test_size, random_state=42, stratify=y_proc['diabetic'])
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1-val_final_ratio, random_state=42, stratify=y_train_val['diabetic'])

        y_scaler = StandardScaler()
        y_train[reg_cols] = y_scaler.fit_transform(y_train[reg_cols])
        y_val[reg_cols] = y_scaler.transform(y_val[reg_cols])
        y_test[reg_cols] = y_scaler.transform(y_test[reg_cols])

        X_train_genes, X_train_context = X_train.iloc[:, :-3], X_train.iloc[:, -3:]
        X_val_genes, X_val_context = X_val.iloc[:, :-3], X_val.iloc[:, -3:]
        X_test_genes, X_test_context = X_test.iloc[:, :-3], X_test.iloc[:, -3:]

        selector = VarianceThreshold(threshold=0.1)
        X_train_g_fs = selector.fit_transform(X_train_genes)
        X_val_g_fs = selector.transform(X_val_genes)
        X_test_g_fs = selector.transform(X_test_genes)

        scaler_g = MinMaxScaler().fit(X_train_g_fs)
        scaler_c = MinMaxScaler().fit(X_train_context)

        return (
            [scaler_g.transform(X_train_g_fs), scaler_g.transform(X_val_g_fs), scaler_g.transform(X_test_g_fs)],
            [scaler_c.transform(X_train_context), scaler_c.transform(X_val_context), scaler_c.transform(X_test_context)],
            [get_target_dict(y_train), get_target_dict(y_val), get_target_dict(y_test)],
            y_scaler
        )
    
    # --- LÓGICA MULTI-ATTENTION ---
    elif architecture_type == 'Multi-Attention':
        X_train_list, X_val_list, X_test_list = [], [], []
        
        first_family = list(X.keys())[0]
        indices = np.arange(len(y_proc))
        idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=42, stratify=y_proc['diabetic'])
        idx_train, idx_val = train_test_split(idx_train_val, test_size=1-val_final_ratio, random_state=42, stratify=y_proc.iloc[idx_train_val]['diabetic'])

        y_train = y_proc.iloc[idx_train].copy()
        y_val = y_proc.iloc[idx_val].copy()
        y_test = y_proc.iloc[idx_test].copy()

        y_scaler = StandardScaler()
        y_train[reg_cols] = y_scaler.fit_transform(y_train[reg_cols])
        y_val[reg_cols] = y_scaler.transform(y_val[reg_cols])
        y_test[reg_cols] = y_scaler.transform(y_test[reg_cols])

        for family, df in X.items():
            genes = df.iloc[:, :-3].values
            g_train, g_val, g_test = genes[idx_train], genes[idx_val], genes[idx_test]
            
            scaler_f = StandardScaler().fit(g_train)
            X_train_list.append(scaler_f.transform(g_train))
            X_val_list.append(scaler_f.transform(g_val))
            X_test_list.append(scaler_f.transform(g_test))

        context = X[first_family].iloc[:, -3:].copy()
        context['gender'] = context['gender'].map({'Male': 1, 'Female': 0})
        c_vals = context.values
        c_train, c_val, c_test = c_vals[idx_train], c_vals[idx_val], c_vals[idx_test]
        
        scaler_ctx = StandardScaler().fit(c_train)
        X_train_list.append(scaler_ctx.transform(c_train))
        X_val_list.append(scaler_ctx.transform(c_val))
        X_test_list.append(scaler_ctx.transform(c_test))

        return (
            X_train_list, X_val_list, X_test_list, 
            get_target_dict(y_train), get_target_dict(y_val), get_target_dict(y_test),
            y_scaler
        )

def autoencoder_architecture(input_size, context_size=3):
    input_genes = layers.Input(shape=(input_size,), name="Transcriptoma")
    input_ctx = layers.Input(shape=(context_size,), name="Entrada_Contexto")

    noise = layers.GaussianNoise(0.05)(input_genes) 

    x = layers.Dense(1024, activation='relu')(noise)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x) 
    
    latent_space = layers.Dense(128, activation='relu', name="Espacio_Latente")(x)

    d = layers.Dense(512, activation='relu')(latent_space)
    d = layers.BatchNormalization()(d)

    d = layers.Dense(1024, activation='relu')(d)
    d = layers.BatchNormalization()(d)

    reconstruction = layers.Dense(input_size, activation='sigmoid', name="Reconstruccion")(d)

    merged = layers.Concatenate()([latent_space, input_ctx])
    
    p = layers.Dense(128, activation='relu')(merged)
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.2)(p) 
    
    p = layers.Dense(64, activation='relu')(p)
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.2)(p) 
    
    # --- CABEZALES REDUCIDOS ---
    out_diabetic = layers.Dense(1, activation='sigmoid', name='diabetic')(p)
    out_hypertensive = layers.Dense(1, activation='sigmoid', name='hypertensive')(p)
    out_hba1c = layers.Dense(1, activation='linear', name='hba1c')(p)
    out_insulin = layers.Dense(1, activation='linear', name='fasting_insulin')(p)

    model = models.Model(
        inputs=[input_genes, input_ctx], 
        outputs=[reconstruction, out_diabetic, out_hypertensive, out_hba1c, out_insulin]
    )

    return model

def autoencoder_train(model, X_train_g, y_train_dict, X_val_g, y_val_dict, X_train_c, X_val_c):
    def prepare_multi_output_dict(genes_scaled, clinical_dict):
        full_targets = clinical_dict.copy()
        full_targets["Reconstruccion"] = genes_scaled
        return full_targets

    y_train_final = prepare_multi_output_dict(X_train_g, y_train_dict)
    y_val_final = prepare_multi_output_dict(X_val_g, y_val_dict)

    # Solo las pérdidas necesarias
    losses = {
        "Reconstruccion": "mse",
        "diabetic": "binary_crossentropy",
        "hypertensive": "binary_crossentropy",
        "hba1c": "mse",
        "fasting_insulin": "mse"
    }

    loss_weights = {
        "Reconstruccion": 0.2,    
        "diabetic": 5.0,           
        "hypertensive": 2.0,
        "hba1c": 2.0,              
        "fasting_insulin": 1.0  # Ancla secundaria de gravedad
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "diabetic": "accuracy",
            'hypertensive': 'accuracy', 
            "hba1c": "mae", 
            'fasting_insulin':'mae'
        }
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    EPOCHS = 200
    BATCH_SIZE = 32 

    history = model.fit(
        x=[X_train_g, X_train_c], 
        y=y_train_final,          
        validation_data=([X_val_g, X_val_c], y_val_final),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1 
    )

    return history

def multi_attention_train(model, X_train_list, y_train_dict, X_val_list, y_val_dict):
    # 1. Definimos el orden EXACTO de las 4 salidas del modelo
    target_names = ["diabetic", "hypertensive", "hba1c", "fasting_insulin"]

    # 2. Convertimos los diccionarios 'y' a listas para Keras
    y_train_list_targets = [y_train_dict[name] for name in target_names]
    y_val_list_targets = [y_val_dict[name] for name in target_names]

    # 3. Definimos las funciones de pérdida en lista (4 elementos)
    loss_list = [
        "binary_crossentropy", # diabetic
        "binary_crossentropy", # hypertensive
        "mse",                 # hba1c
        "mse"                  # fasting_insulin
    ]

    # 4. Pesos de Pérdida en lista
    loss_weights_list = [
        5.0, # diabetic (prioridad máxima)
        2.0, # hypertensive 
        2.0, # hba1c (ancla de gravedad)
        1.0  # fasting_insulin (ancla secundaria)
    ]

    # 5. Métricas para cada cabezal en lista
    metrics_list = [
        ['accuracy'], # diabetic
        ['accuracy'], # hypertensive
        ['mae'],      # hba1c
        ['mae']       # fasting_insulin
    ]

    # 6. Compilación
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss=loss_list,
        loss_weights=loss_weights_list,
        metrics=metrics_list
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # 7. CÁLCULO DE SAMPLE WEIGHTS 
    pesos_d = compute_class_weight('balanced', classes=np.unique(y_train_dict['diabetic']), y=y_train_dict['diabetic'])
    pesos_muestras_diabetes = np.where(y_train_dict['diabetic'] == 1, pesos_d[1], pesos_d[0])

    pesos_h = compute_class_weight('balanced', classes=np.unique(y_train_dict['hypertensive']), y=y_train_dict['hypertensive'])
    pesos_muestras_hiper = np.where(y_train_dict['hypertensive'] == 1, pesos_h[1], pesos_h[0])

    pesos_neutros = np.ones_like(pesos_muestras_diabetes)

    pesos_completos_dict = {
        "diabetic": pesos_muestras_diabetes,
        "hypertensive": pesos_muestras_hiper,
        "hba1c": pesos_neutros,
        "fasting_insulin": pesos_neutros
    }
    
    # Transformamos a lista en el orden exacto para Keras (4 elementos)
    sample_weight_list = [pesos_completos_dict[name] for name in target_names]

    EPOCHS = 200
    BATCH_SIZE = 32

    print("\nIniciando entrenamiento Clínico...")
    history = model.fit(
        x=X_train_list, 
        y=y_train_list_targets,           
        validation_data=(X_val_list, y_val_list_targets), 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sample_weight_list, 
        callbacks=[early_stop, reduce_lr],
        verbose=1 
    )

    return history

def parallel_autoencoders_architecture(input_shapes, EMBEDDING_DIM=64):
    inputs = []
    outputs = []
    nombres_familias = ["Hydrolases", "Isomerases", "Ligases", "Lyases", "Oxidoreductases", "Transferases", "Translocases"]

    for i in range(7):
        inp = layers.Input(shape=(input_shapes[i],), name=f"in_{nombres_familias[i]}")
        inputs.append(inp)
        
        noise = layers.GaussianNoise(0.05)(inp)
        x = layers.Dense(256, activation='relu')(noise)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        latent = layers.Dense(EMBEDDING_DIM, activation='relu', name=f"Latent_{nombres_familias[i]}")(x)
        latent = layers.BatchNormalization()(latent)
        
        d = layers.Dense(256, activation='relu')(latent)
        d = layers.BatchNormalization()(d)
        
        out = layers.Dense(input_shapes[i], activation='linear', name=f"out_{nombres_familias[i]}")(d)
        outputs.append(out)

    parallel_ae = models.Model(inputs=inputs, outputs=outputs, name="Parallel_Autoencoders")
    return parallel_ae

def parallel_autoencoders_train(model, X_train_list, X_val_list):
    # Toma solo los primeros 7 elementos (genes), ignora el contexto (índice 7)
    X_train_genes = X_train_list[:7]
    X_val_genes = X_val_list[:7]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    print("\n[+] FASE 1: Entrenamiento de Autoencoders Paralelos...")
    history = model.fit(
        x=X_train_genes, y=X_train_genes, 
        validation_data=(X_val_genes, X_val_genes),
        epochs=150, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=1
    )
    return history

def multi_attention_transformer_architecture(input_shapes, trained_ae, context_size=3):
    in_hydro = layers.Input(shape=(input_shapes[0],), name="ma_in_Hydrolases")
    in_iso = layers.Input(shape=(input_shapes[1],), name="ma_in_Isomerases")
    in_ligases = layers.Input(shape=(input_shapes[2],), name="ma_in_Ligases")
    in_lyases = layers.Input(shape=(input_shapes[3],), name="ma_in_Lyases")
    in_oxi = layers.Input(shape=(input_shapes[4],), name="ma_in_Oxidoreductases")
    in_transf = layers.Input(shape=(input_shapes[5],), name="ma_in_Transferases")
    in_transl = layers.Input(shape=(input_shapes[6],), name="ma_in_Translocases")
    in_ctx = layers.Input(shape=(context_size,), name="Entrada_Contexto")

    inputs_familias = [in_hydro, in_iso, in_ligases, in_lyases, in_oxi, in_transf, in_transl]
    nombres_familias = ["Hydrolases", "Isomerases", "Ligases", "Lyases", "Oxidoreductases", "Transferases", "Translocases"]
    
    emb_familias = []
    
    # 1. EXTRACCIÓN Y FINE-TUNING
    for i in range(7):
        extractor = models.Model(
            inputs=trained_ae.input[i], 
            outputs=trained_ae.get_layer(f"Latent_{nombres_familias[i]}").output
        )
        extractor.trainable = True # Fine-Tuning activado
        emb_familias.append(extractor(inputs_familias[i]))

    EMBEDDING_DIM = 64
    r_familias = [layers.Reshape((1, EMBEDDING_DIM))(emb) for emb in emb_familias]
    stacked_families = layers.Concatenate(axis=1)(r_familias)

    # 2. CAPA DE ATENCIÓN MULTI-HEAD
    mha_output, attention_scores = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name="MHA_Familias"
    )(query=stacked_families, value=stacked_families, key=stacked_families, return_attention_scores=True)

    mha_added = layers.Add()([stacked_families, mha_output])
    mha_norm = layers.LayerNormalization()(mha_added)

    # 3. FEED-FORWARD NETWORK (FFN)
    ffn_output = layers.Dense(EMBEDDING_DIM, activation='relu')(mha_norm)
    ffn_added = layers.Add()([mha_norm, ffn_output])
    ffn_norm = layers.LayerNormalization()(ffn_added)

    # 4. APLANAR EN LUGAR DE PROMEDIAR (Preservar la identidad de cada familia)
    metabolic_profile = layers.Flatten()(ffn_norm)

    # 5. RAMA CLÍNICA
    merged = layers.Concatenate()([metabolic_profile, in_ctx])

    p = layers.Dense(128, activation='relu')(merged) 
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.3)(p) 
    
    p = layers.Dense(64, activation='relu')(p) 
    p = layers.BatchNormalization()(p)
    p = layers.Dropout(0.3)(p) 

    out_diabetic = layers.Dense(1, activation='sigmoid', name='diabetic')(p)
    out_hypertensive = layers.Dense(1, activation='sigmoid', name='hypertensive')(p)
    out_hba1c = layers.Dense(1, activation='linear', name='hba1c')(p)
    out_insulin = layers.Dense(1, activation='linear', name='fasting_insulin')(p)

    model = models.Model(
        inputs=inputs_familias + [in_ctx],
        outputs=[out_diabetic, out_hypertensive, out_hba1c, out_insulin]
    )

    return model

def evaluate_model_metrics(model, X_test_inputs, y_test, y_scaler, name):
    predicciones_test = model.predict(X_test_inputs)

    is_autoencoder = (name == 'Denoising Autoencoder')
    
    # Ajustar índices a las nuevas 4 (+1) salidas
    if is_autoencoder:
        idx_diabetic = 1
        idx_hypertensive = 2
        idx_hba1c = 3
        idx_insulin = 4
    else:
        idx_diabetic = 0
        idx_hypertensive = 1
        idx_hba1c = 2
        idx_insulin = 3

    pred_diabetes_prob = predicciones_test[idx_diabetic] 
    pred_diabetes_clase = (pred_diabetes_prob > 0.5).astype(int)
    y_test_diab_real = y_test['diabetic']

    pred_hipertension_prob = predicciones_test[idx_hypertensive]
    pred_hipertension_clase = (pred_hipertension_prob > 0.5).astype(int)
    y_test_hiper_real = y_test['hypertensive']

    cm_diab = confusion_matrix(y_test_diab_real, pred_diabetes_clase)
    tn, fp, fn, tp = cm_diab.ravel() 
    sensibilidad_diab = tp / (tp + fn) 
    especificidad_diab = tn / (tn + fp) 

    cm_hyper = confusion_matrix(y_test_hiper_real, pred_hipertension_clase)
    tn, fp, fn, tp = cm_hyper.ravel() 
    sensibilidad_hyper = tp / (tp + fn) 
    especificidad_hyper = tn / (tn + fp) 

    print(f"\n--- MÉTRICAS CLÍNICAS DIABETES ({name}) ---")
    print(f"Sensibilidad (Recall clase 1): {sensibilidad_diab:.2f}")
    print(f"Especificidad (Recall clase 0): {especificidad_diab:.2f}")
    print("\nReporte Completo:")
    print(classification_report(y_test_diab_real, pred_diabetes_clase, target_names=['No Diabético', 'Diabético']))

    print(f"\n--- MÉTRICAS CLÍNICAS HIPERTENSIÓN ({name}) ---")
    print(f"Sensibilidad (Recall clase 1): {sensibilidad_hyper:.2f}")
    print(f"Especificidad (Recall clase 0): {especificidad_hyper:.2f}")
    print("\nReporte Completo:")
    print(classification_report(y_test_hiper_real, pred_hipertension_clase, target_names=['No Hipertenso', 'Hipertenso']))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_diab, annot=True, fmt='d', cmap='Blues', ax=axs[0],
                xticklabels=['Predicción: Sano', 'Predicción: Diabético'], 
                yticklabels=['Real: Sano', 'Real: Diabético'])
    axs[0].set_title(f'Matriz de Confusión - Diabetes - {name}')

    sns.heatmap(cm_hyper, annot=True, fmt='d', cmap='Blues', ax=axs[1],
                xticklabels=['Predicción: Sano', 'Predicción: Hipertenso'], 
                yticklabels=['Real: Sano', 'Real: Hipertenso'])
    axs[1].set_title(f'Matriz de Confusión - Hipertensión - {name}')

    plt.tight_layout()
    plt.show()

    # Evaluacion Exclusiva para HbA1c e Insulina
    nombres_reg = ['hba1c', 'fasting_insulin']

    preds_reg_scaled = np.column_stack([predicciones_test[idx_hba1c], predicciones_test[idx_insulin]])
    reales_reg_scaled = np.column_stack([y_test[col] for col in nombres_reg])

    preds_real = y_scaler.inverse_transform(preds_reg_scaled)
    reales_real = y_scaler.inverse_transform(reales_reg_scaled)

    resultados_regresion = []

    for i, col_name in enumerate(nombres_reg):
        y_true_col = reales_real[:, i]
        y_pred_col = preds_real[:, i]
        
        mae = mean_absolute_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        r2 = r2_score(y_true_col, y_pred_col)
        
        resultados_regresion.append({
            'Variable': col_name,
            'MAE (Error Absoluto Promedio)': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R^2 Score': round(r2, 3)
        })

    df_resultados_reg = pd.DataFrame(resultados_regresion)
    print(f"\n--- DESEMPEÑO DEL MODELO EN VARIABLES CONTINUAS PARA {name}---")
    print(df_resultados_reg.to_string(index=False))

    return df_resultados_reg

def plot_history(history):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Val Loss')
    axs[0].set_title('Pérdida Total Combinada')
    axs[0].legend()
    
    axs[1].plot(history.history['diabetic_accuracy'], label='Train Diab Acc')
    axs[1].plot(history.history['val_diabetic_accuracy'], label='Val Diab Acc')
    axs[1].set_title('Precisión en Predicción de Diabetes')
    axs[1].legend()

    axs[2].plot(history.history['hypertensive_accuracy'], label='Train Hiper Acc')
    axs[2].plot(history.history['val_hypertensive_accuracy'], label='Val Hiper Acc')
    axs[2].set_title('Precisión en Predicción de Hipertensión')
    axs[2].legend()
    
    plt.show()

def plot_comparative_roc(model_ae, model_multi, X_test_ae, X_test_multi, y_test):
    print("\n[+] Generando Análisis Comparativo ROC...")
    
    # 1. Extraer las probabilidades reales de la clase positiva (Diabetes)
    # Para el DAE original, la diabetes está en el índice 1
    preds_ae = model_ae.predict(X_test_ae)
    prob_diab_ae = preds_ae[1].ravel() 
    
    # Para el Transformer, la diabetes está en el índice 0
    preds_multi = model_multi.predict(X_test_multi)
    prob_diab_multi = preds_multi[0].ravel()
    
    # Etiquetas reales
    y_real = y_test['diabetic']
    
    # 2. Calcular FPR (Falsos Positivos) y TPR (Sensibilidad) para ambos
    fpr_ae, tpr_ae, thresholds_ae = roc_curve(y_real, prob_diab_ae)
    auc_ae = auc(fpr_ae, tpr_ae)
    
    fpr_multi, tpr_multi, thresholds_multi = roc_curve(y_real, prob_diab_multi)
    auc_multi = auc(fpr_multi, tpr_multi)
    
    # 3. Construir el gráfico de alta calidad
    plt.figure(figsize=(9, 7))
    
    # Línea del Autoencoder Global
    plt.plot(fpr_ae, tpr_ae, color='#E67E22', lw=2.5, 
            label=f'Autoencoder Global (AUC = {auc_ae:.3f})')
            
    # Línea del Transformer Híbrido
    plt.plot(fpr_multi, tpr_multi, color='#2980B9', lw=2.5, 
            label=f'Transformer por Familias (AUC = {auc_multi:.3f})')
    
    # Línea de azar (Peor escenario)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Azar (AUC = 0.500)')
    
    # Formateo estético para artículos científicos
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12, fontweight='bold')
    plt.title('Comparación de Curvas ROC - Diagnóstico de DM2', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.show()
    
    return auc_ae, auc_multi

# ====================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ====================================================================

# Load an preprocess all the datasets
data_transcriptomic, data_clinic, data_genes = load_preprocess_data()

# Split in train, test and validation the dataset fot both architectures. 
genes_data_ae, context_data_ae, targets_ae, y_scaler_ae = train_test_validation_split(data_transcriptomic, data_clinic, 'Autoencoder', 0.15)
X_train_multi, X_val_multi, X_test_multi, y_train_multi, y_val_multi, y_test_multi, y_scaler_multi = train_test_validation_split(data_genes, data_clinic, 'Multi-Attention', 0.15)

# Desempaquetar los resultados
X_train_g, X_val_g, X_test_g = genes_data_ae
X_train_c, X_val_c, X_test_c = context_data_ae
y_train, y_val, y_test = targets_ae

# --- ENTRENAMIENTO DEL MODELO ORIGINAL (Denoising Autoencoder) ---
num_genes = genes_data_ae[0][0].shape[0]
model_ae = autoencoder_architecture(num_genes, context_size=3)
history_ae = autoencoder_train(model_ae, X_train_g, y_train, X_val_g, y_val, X_train_c, X_val_c)

# Evaluación del AE Original
results_reg_ae = evaluate_model_metrics(model_ae, [X_test_g, X_test_c], y_test, y_scaler_ae, 'Denoising Autoencoder')
plot_history(history_ae)

# --- ENTRENAMIENTO DEL NUEVO PIPELINE (Autoencoders Paralelos + Transformer) ---
input_shapes = [X_train_multi[i].shape[1] for i in range(7)] 

# Fase 1: Entrenamiento de los Autoencoders Paralelos
model_ae_paralelo = parallel_autoencoders_architecture(input_shapes, EMBEDDING_DIM=64)
history_ae_paralelo = parallel_autoencoders_train(model_ae_paralelo, X_train_multi, X_val_multi)

# Fase 2: Construcción y Entrenamiento del Transformer (usando Fine-Tuning + FFN + Flatten)
model_multi_transformer = multi_attention_transformer_architecture(input_shapes, model_ae_paralelo, context_size=3)

# Aprovechamos tu misma función de entrenamiento original para el transformer
history_multi_transformer = multi_attention_train(model_multi_transformer, X_train_multi, y_train_multi, X_val_multi, y_val_multi)

# Evaluación del Transformer Multi-Attention
results_reg_multi = evaluate_model_metrics(model_multi_transformer, X_test_multi, y_test_multi, y_scaler_multi, 'Transformer Multi-Attention')
plot_history(history_multi_transformer)

auc_ae, auc_multi = plot_comparative_roc(
    model_ae=model_ae, 
    model_multi=model_multi_transformer, 
    X_test_ae=[X_test_g, X_test_c], 
    X_test_multi=X_test_multi, 
    y_test=y_test
)