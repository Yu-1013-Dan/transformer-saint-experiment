# All imports should be at the top of the script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib # For saving preprocessors
import traceback # For detailed error printing
import os # For creating output directory

# --- Configuration & Parameters ---
# !!! USER: YOU MUST ADJUST THESE BASED ON YOUR FINAL DEVICE CLASS MAPPING AND FEATURE ANALYSIS !!!

TARGET_COLUMN = 'target_device_type'
# USER CONFIRMED: 'device_mac' column contains the raw device string names
RAW_DEVICE_NAME_SOURCE_COLUMN = 'device_mac'

# Directory to save processed data and preprocessors
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Features to drop outright
ID_FEATURES_TO_DROP = ['stream', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip',
                       RAW_DEVICE_NAME_SOURCE_COLUMN # Dropped from X features after target creation
                      ]

# Features that are complex or initially deprioritized for lightweight model
COMPLEX_OR_LOW_PRIORITY_FEATURES_TO_DROP = [
    'handshake_ciphersuites', # List processing is complex
    'user_agent',             # Very high cardinality
    'http_uri',               # Very high cardinality
    'eth_dst_oui',            # Destination OUI less stable than source
    'icmp_type', 'icmp_checksum_status', 'icmp_data_size', # ICMP related
    'most_freq_spot', 'min_et', 'q1', 'min_e', 'var_e', 'q1_e', # Meaning unclear or not core
    'time_since_previously_displayed_frame',
    # Drop all other time-windowed stats not explicitly selected to reduce redundancy and aid lightweighting
    # The SELECTED_NUMERICAL_FEATURES will list the specific ones we keep.
    # Listing all others to drop is too verbose; selection is better.
]

# --- EXPERT SELECTED FEATURE LISTS (User should verify and refine) ---
SELECTED_CATEGORICAL_FEATURES = [
    'eth_src_oui', 'port_class_dst', 'protocol', 'highest_layer',
    'handshake_version', 'tls_server', 'http_host', 'http_request_method',
    'dns_query_type'
]
SELECTED_NUMERICAL_FEATURES = [
    'ttl', 'payload_entropy', 'tcp_window_size',
    'handshake_cipher_suites_length', 'handshake_extensions_length',
    'dns_len_qry', 'dns_len_ans', 'dns_interval', 'l3_ip_dst_count',
    'sum_p', 'min_p', 'max_p', 'med_p', 'average_p', 'var_p', 'iqr_p', # Packet size stats
    'inter_arrival_time', 'jitter',
    'stream_5_count', 'stream_5_mean',
    'stream_60_count', 'stream_60_mean',
    'stream_jitter_5_mean', 'stream_jitter_60_mean',
    'src_ip_5_count', 'src_ip_60_count',
    'ntp_interval'
]
# Ensure all selected features are unique and 'protocol' (if engineered) is included
ALL_SELECTED_FEATURES = list(set(SELECTED_CATEGORICAL_FEATURES + SELECTED_NUMERICAL_FEATURES + ['protocol']))


# --- Helper Functions ---

def load_and_combine_data(filepath_list, nrows_per_file=None):
    """Loads data from a list of CSV files and combines them."""
    print(f"开始从列表加载和合并数据文件: {filepath_list} (Starting to load and combine data files from list: {filepath_list})")
    all_dfs = []
    for filepath in filepath_list:
        print(f"正在从 {filepath} 加载数据 (Loading data from {filepath})...")
        try:
            df_temp = pd.read_csv(filepath, nrows=nrows_per_file)
            if df_temp.columns[0].startswith('Unnamed:'):
                print(f"丢弃第一个未命名索引列: {df_temp.columns[0]} (Dropping first unnamed index column: {df_temp.columns[0]})")
                df_temp = df_temp.iloc[:, 1:]
            all_dfs.append(df_temp)
            print(f"已添加 {filepath} 到待合并列表。(Added {filepath} to list for concatenation.)")
        except FileNotFoundError:
            print(f"警告: 文件 {filepath} 未找到，已跳过。(Warning: File {filepath} not found, skipping.)")
        except Exception as e:
            print(f"加载文件 {filepath} 时出错: {e} (Error loading file {filepath}: {e})")

    if not all_dfs:
        raise ValueError("没有成功加载任何数据文件，请检查文件路径列表。(No data files were successfully loaded. Please check the file path list.)")

    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"所有可加载的数据文件已合并。合并后的原始形状: {df_combined.shape} (All loadable data files combined. Combined raw shape: {df_combined.shape})")
    return df_combined


def create_and_filter_target(df_input, raw_device_name_source_col, target_col_name):
    """
    Creates the target variable by mapping raw device names to final classes,
    filters for selected smart home devices, and removes classes with too few samples.
    """
    print(f"正在使用源列 '{raw_device_name_source_col}' 创建目标变量 '{target_col_name}' 并筛选设备 (Creating target variable '{target_col_name}' using source column '{raw_device_name_source_col}' and filtering devices)...")
    df = df_input.copy()

    # 1. Define your device name to target class label mapping dictionary
    # !!! THIS IS THE MOST CRITICAL PART FOR YOU TO CUSTOMIZE !!!
    # Build this map based on your analysis of df[raw_device_name_source_col].value_counts()
    device_to_target_class_map = {
        # Cameras
        'Arlo Q Indoor Camera': 'Camera_Arlo_Q',
        'Nest Indoor Camera': 'Camera_Nest_Indoor',
        'Yi Indoor 2 Camera': 'Camera_Yi_Indoor', 'Yi Indoor Camera': 'Camera_Yi_Indoor',
        'Yi Outdoor Camera': 'Camera_Yi_Outdoor',
        'Wyze Camera': 'Camera_Wyze',
        'Home Eye Camera': 'Camera_Home_Eye',
        'Netatmo Camera': 'Camera_Netatmo',
        'Rbcior Camera': 'Camera_Rbcior',
        'HeimVision Smart WiFi Camera': 'Camera_HeimVision',
        'TP-Link Tapo Camera': 'Camera_TP_Link_Tapo',
        'AMCREST WiFi Camera': 'Camera_AMCREST',
        'DCS8000LHA1 D-Link Mini Camera': 'Camera_DLink_Mini',

        # Speakers
        'Amazon Echo Show': 'Speaker_Amazon_Echo_Show',
        'Amazon Echo Dot 2': 'Speaker_Amazon_Echo_Dot', 'Amazon Echo Dot 1': 'Speaker_Amazon_Echo_Dot',
        'Amazon Echo Studio': 'Speaker_Amazon_Echo_Studio',
        'Google Nest Mini Speaker': 'Speaker_Google_Nest_Mini',
        'harman kardon (Ampak Technology)': 'Speaker_Harman_Kardon',
        'Sonos One Speaker': 'Speaker_Sonos_One',

        # Hubs/Bridges
        'SmartThings Hub': 'Hub_SmartThings',
        'Philips Hue Bridge': 'Hub_Philips_Hue',
        'AeoTec Smart Home Hub': 'Hub_AeoTec',
        'Eufy HomeBase 2': 'Hub_Eufy_HomeBase',
        'Arlo Base Station': 'Hub_Arlo_Base_Station',

        # Plugs/Strips
        'GoSund Smart Plug WP3 (1)': 'Plug_GoSund', 'Gosund Smart Plug WP3 (2)': 'Plug_GoSund',
        'GoSund Smart Plug WP2 (2)': 'Plug_GoSund', 'GoSund Smart plug WP2 (1)': 'Plug_GoSund',
        'GoSund Smart plug WP2 (3)': 'Plug_GoSund',
        'Gosund Power strip (1)': 'PowerStrip_Gosund', 'GoSund Power strip (2)': 'PowerStrip_Gosund',
        'Yutron Plug 2': 'Plug_Yutron', 'Yutron Plug 1': 'Plug_Yutron',
        'Teckin Plug 2': 'Plug_Teckin', 'Teckin Plug 1': 'Plug_Teckin',
        'Amazon Plug': 'Plug_Amazon',
        'Wemo smart plug 2 (Wemo id: Wemo.Mini.4A3)': 'Plug_Wemo',
        'Wemo smart plug 1 (Wemo id: Wemo.Mini.AD3)': 'Plug_Wemo',

        # Lighting
        'GoSund Bulb': 'Bulb_GoSund',
        'LampUX RGB': 'Lighting_LampUX_RGB',
        'HeimVision SmartLife Radio/Lamp': 'Lamp_HeimVision_Radio',
        'Lumiman bulb': 'Bulb_Lumiman',
        'Teckin Light Strip': 'Lighting_Teckin_Strip',
        'LIFX Lightbulb': 'Bulb_LIFX',

        # Appliances/Sensors
        'Netatmo Weather Station': 'Sensor_Netatmo_Weather',
        'Atomi Coffee Maker': 'Appliance_Atomi_Coffee',
        'Cocoon Smart HVAC Fan': 'Appliance_Cocoon_HVAC_Fan',
        'Levoit Air Purifier': 'Appliance_Levoit_Air_Purifier',
        'iRobot Roomba': 'Appliance_iRobot_Roomba',
        'Govee Smart Humidifer': 'Appliance_Govee_Humidifier',
        
        # TV
        'LG Smart TV': 'TV_LG_Smart',
    }
    
    if raw_device_name_source_col not in df.columns:
        raise ValueError(f"原始设备名源列 '{raw_device_name_source_col}' 在DataFrame中未找到。可用列: {df.columns.tolist()} (Raw device name source column '{raw_device_name_source_col}' not found in DataFrame. Available columns: {df.columns.tolist()})")

    df.loc[:, target_col_name] = df[raw_device_name_source_col].map(device_to_target_class_map)

    original_row_count = len(df)
    df.dropna(subset=[target_col_name], inplace=True)
    rows_dropped_unmapped = original_row_count - len(df)
    if rows_dropped_unmapped > 0:
        print(f"丢弃了 {rows_dropped_unmapped} 行未被映射到任何目标类别的记录 (Dropped {rows_dropped_unmapped} rows that were not mapped to any target class).")

    if df.empty:
        raise ValueError("在应用device_map后DataFrame为空。请检查映射字典的键与源列中的唯一值，或确认是否有设备被映射。(DataFrame is empty after applying device_map. Check map keys against unique values in the source column or if all devices were unmapped.)")

    min_samples_per_class = 1000  # 用户: 根据需要调整此阈值 (e.g., 500 or 1000)
    class_counts = df[target_col_name].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples_per_class].index.tolist()

    if not classes_to_keep:
        raise ValueError(f"没有类别满足最小样本阈值 {min_samples_per_class}。请考虑降低阈值、检查数据或重新评估映射。(No classes meet the minimum sample threshold of {min_samples_per_class}. Consider a lower threshold, check data, or re-evaluate mappings.)")

    original_row_count_before_min_sample_filter = len(df)
    df = df[df[target_col_name].isin(classes_to_keep)]
    rows_dropped_low_sample = original_row_count_before_min_sample_filter - len(df)
    if rows_dropped_low_sample > 0:
        print(f"从样本数少于 {min_samples_per_class} 的类别中丢弃了 {rows_dropped_low_sample} 行记录 (Dropped {rows_dropped_low_sample} rows from classes with fewer than {min_samples_per_class} samples).")
    
    if df.empty:
        raise ValueError("按最小样本数筛选后DataFrame为空。没有类别满足阈值。(DataFrame is empty after filtering by min_samples_per_class. No classes met the threshold.)")

    print(f"目标变量创建和筛选完成。保留了 {len(classes_to_keep)} 个类别: {classes_to_keep} (Target variable created and filtered. Kept {len(classes_to_keep)} classes: {classes_to_keep})")
    print(f"目标创建/筛选后的最终形状: {df.shape} (Final shape after target creation/filtering: {df.shape})")
    print(f"最终类别分布:\n{df[target_col_name].value_counts()} (Final class distribution:\n{df[target_col_name].value_counts()})")

    return df


def initial_clean_and_type_convert(df_features_input):
    """对选定特征进行基础清洗和类型转换 (Basic cleaning and type conversions for selected features)."""
    print("正在对特征进行初步清洗和类型转换 (Performing initial cleaning and type conversion on features)...")
    df = df_features_input.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            # Handle specific 'none' strings if they mean missing for certain columns.
            # For example, if 'tls_server' or 'http_host' can have 'none' as a meaningful category, leave it.
            # If 'none' means NaN for specific columns (e.g. http_response_code being 'none'), replace it.
            # This needs careful consideration based on dataset documentation.
            # For now, we assume 'none' is a potential category and will be handled by imputation or Top-N.

    for col in ['l4_tcp', 'l4_udp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    if 'handshake_version' in df.columns:
        # Treat -1 as a valid category, NaNs (if any after string 'none' replacement) as another
        df['handshake_version'] = pd.to_numeric(df['handshake_version'], errors='coerce').fillna(-2).astype(str)

    # Convert all selected numerical features to numeric, coercing errors
    # This is important for columns that might have mixed types or string NaNs
    # Iterate only over columns present in the DataFrame
    for col in [num_col for num_col in SELECTED_NUMERICAL_FEATURES if num_col in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer_features(df_features_input):
    """工程化新特征或转换现有特征 (Engineer new features or transform existing ones)."""
    print("正在工程化特征 (Engineering features)...")
    df = df_features_input.copy()
    if 'l4_tcp' in df.columns and 'l4_udp' in df.columns:
        conditions = [(df['l4_tcp'] == 1), (df['l4_udp'] == 1)]
        choices = ['TCP', 'UDP']
        df['protocol'] = np.select(conditions, choices, default='Other')
    else:
        df['protocol'] = 'Unknown' # Should not happen if l4_tcp/udp are present
    return df

def handle_high_cardinality_categorical(df_train_input, df_val_input, df_test_input, column, top_n=30):
    """通过保留Top-N + 'OTHER_CAT'来处理指定列的高基数类别 (Handles high cardinality for a specific column by keeping Top-N + 'OTHER_CAT')."""
    print(f"正在处理列 '{column}' 的高基数类别 (Top-{top_n}) (Handling high cardinality for column: {column} (Top-{top_n}))")
    # Work on copies to avoid SettingWithCopyWarning
    df_train, df_val, df_test = df_train_input.copy(), df_val_input.copy(), df_test_input.copy()

    # Calculate Top-N based on training data only, ensure it's string for value_counts
    top_n_categories = df_train[column].astype(str).value_counts().nlargest(top_n).index.tolist()
    
    if not top_n_categories: # Handle case where column might be all NaNs or very few values
        print(f"警告: 训练数据中列 '{column}' 未找到Top类别。可能该列值单一或多为NaN。跳过此列的Top-N处理。(Warning: No top categories found for {column} in training data. Column might be monotonous or mostly NaN. Skipping Top-N for it.)")
        # Ensure column is string type even if not processed by Top-N for consistent imputation later
        for df_set in [df_train, df_val, df_test]:
            df_set[column] = df_set[column].astype(str)
        return df_train, df_val, df_test

    for df_set in [df_train, df_val, df_test]:
        df_set.loc[:, column] = df_set[column].astype(str).apply(lambda x: x if x in top_n_categories else 'OTHER_CAT')
    return df_train, df_val, df_test

# --- 主要预处理流程 (Main Preprocessing Pipeline) ---
def main_preprocess_pipeline(filepath_list, nrows_per_file=None):
    """主要预处理流程 (Main preprocessing pipeline)."""
    # 1. 加载并合并数据
    df_raw = load_and_combine_data(filepath_list, nrows_per_file=nrows_per_file)

    # 2. 定义目标变量并筛选样本
    df_targeted = create_and_filter_target(df_raw, RAW_DEVICE_NAME_SOURCE_COLUMN, TARGET_COLUMN)
    
    y = df_targeted[TARGET_COLUMN]
    X = df_targeted.drop(columns=[TARGET_COLUMN], errors='ignore')

    # 3. 初步清洗与类型转换
    X = initial_clean_and_type_convert(X)

    # 4. 丢弃不需要的ID、复杂、低优先级特征
    cols_to_drop_for_sure = ID_FEATURES_TO_DROP + COMPLEX_OR_LOW_PRIORITY_FEATURES_TO_DROP
    # Ensure RAW_DEVICE_NAME_SOURCE_COLUMN (device_mac) is in ID_FEATURES_TO_DROP
    if RAW_DEVICE_NAME_SOURCE_COLUMN not in cols_to_drop_for_sure:
         cols_to_drop_for_sure.append(RAW_DEVICE_NAME_SOURCE_COLUMN)

    X = X.drop(columns=[col for col in cols_to_drop_for_sure if col in X.columns], errors='ignore')
    print(f"丢弃ID、复杂和低优先级特征后。X 形状: {X.shape} (Dropped ID, complex, and low_priority features. X shape: {X.shape})")

    # 5. 工程化特征
    X = engineer_features(X)

    # 6. 仅保留专家选择的特征
    # Make sure 'protocol' is in ALL_SELECTED_FEATURES if it was engineered and meant to be kept
    if 'protocol' in X.columns and 'protocol' not in ALL_SELECTED_FEATURES:
         ALL_SELECTED_FEATURES.append('protocol')
    current_selected_features = [f for f in ALL_SELECTED_FEATURES if f in X.columns]
    
    X = X[current_selected_features].copy() # Use .copy()
    print(f"应用专家特征选择后。X 形状: {X.shape} (Applied expert feature selection. X shape: {X.shape})")
    
    final_categorical_features = [f for f in SELECTED_CATEGORICAL_FEATURES if f in X.columns]
    final_numerical_features = [f for f in SELECTED_NUMERICAL_FEATURES if f in X.columns]
    
    if not current_selected_features: # Check if any features remain AT ALL
        raise ValueError("特征选择后特征集为空。请检查您的特征列表和数据。(Feature set is empty after selection. Check your feature lists and data.)")
    if not final_categorical_features and not final_numerical_features:
        print("警告: 最终类别特征和数值特征列表都为空。模型可能没有输入特征。(Warning: Both final categorical and numerical feature lists are empty. Model might have no input features.)")


    print(f"\n最终选择的类别特征 ({len(final_categorical_features)}): {final_categorical_features} (Final selected categorical features ({len(final_categorical_features)}): {final_categorical_features})")
    print(f"最终选择的数值特征 ({len(final_numerical_features)}): {final_numerical_features[:10]}... (前10个) (Final selected numerical features ({len(final_numerical_features)}): {final_numerical_features[:10]}... (first 10))")

    # 7. 划分数据
    # Splitting into 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y 
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp 
    )
    del X_temp, y_temp # Free up memory
    print(f"\n数据划分: 训练集 ({X_train.shape}), 验证集 ({X_val.shape}), 测试集 ({X_test.shape}) (Data split: Train ({X_train.shape}), Val ({X_val.shape}), Test ({X_test.shape}))")

    # Create copies to avoid SettingWithCopyWarning on slices
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # 8. 处理高基数类别特征
    high_card_cols_to_process = ['tls_server', 'http_host'] # USER: Add 'dns_server' etc. if selected and high card
    for col in high_card_cols_to_process:
        if col in final_categorical_features:
            X_train, X_val, X_test = handle_high_cardinality_categorical(X_train, X_val, X_test, col, top_n=30)

    # 9. 填充缺失值
    print("\n正在填充缺失值 (Imputing missing values)...")
    num_imputer = SimpleImputer(strategy='median')
    if final_numerical_features:
        X_train.loc[:, final_numerical_features] = num_imputer.fit_transform(X_train[final_numerical_features])
        X_val.loc[:, final_numerical_features] = num_imputer.transform(X_val[final_numerical_features])
        X_test.loc[:, final_numerical_features] = num_imputer.transform(X_test[final_numerical_features])

    missing_placeholder = 'MISSING_CAT'
    for col in final_categorical_features:
        X_train.loc[:, col] = X_train[col].astype(str).fillna(missing_placeholder)
        X_val.loc[:, col] = X_val[col].astype(str).fillna(missing_placeholder)
        X_test.loc[:, col] = X_test[col].astype(str).fillna(missing_placeholder)

    # 10. 编码类别特征
    print("\n正在编码类别特征 (Encoding categorical features)...")
    label_encoders = {}
    for col in final_categorical_features:
        le = LabelEncoder()
        all_possible_values = pd.concat([X_train[col], X_val[col], X_test[col]], ignore_index=True).astype(str).unique()
        le.fit(all_possible_values)
        
        X_train.loc[:, col] = le.transform(X_train[col].astype(str))
        X_val.loc[:, col] = le.transform(X_val[col].astype(str))
        X_test.loc[:, col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # 11. 缩放数值特征
    print("\n正在缩放数值特征 (Scaling numerical features)...")
    scaler = StandardScaler()
    if final_numerical_features:
        X_train.loc[:, final_numerical_features] = scaler.fit_transform(X_train[final_numerical_features])
        X_val.loc[:, final_numerical_features] = scaler.transform(X_val[final_numerical_features])
        X_test.loc[:, final_numerical_features] = scaler.transform(X_test[final_numerical_features])
    
    print("\n预处理完成。(Preprocessing complete.)")
    print(f"最终 X_train 形状: {X_train.shape} (Final X_train shape: {X_train.shape})")

    # --- 保存处理后的数据和预处理器 (Save processed data and preprocessors) ---
    print(f"\n正在保存预处理器到 '{OUTPUT_DIR}'... (Saving preprocessors to '{OUTPUT_DIR}'...)")
    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.joblib'))
    if final_numerical_features:
        joblib.dump(num_imputer, os.path.join(OUTPUT_DIR, 'num_imputer.joblib'))
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    print("预处理器已保存。(Preprocessors saved.)")

    print(f"\n正在保存处理后的数据到 '{OUTPUT_DIR}'... (Saving processed data to '{OUTPUT_DIR}'...)")
    try:
        X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_processed.csv'), index=False)
        X_val.to_csv(os.path.join(OUTPUT_DIR, 'X_val_processed.csv'), index=False)
        X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test_processed.csv'), index=False)
        y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train_processed.csv'), index=False, header=True)
        y_val.to_csv(os.path.join(OUTPUT_DIR, 'y_val_processed.csv'), index=False, header=True)
        y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test_processed.csv'), index=False, header=True)
        print("处理后的数据已成功保存到CSV文件。(Processed data successfully saved to CSV files.)")
    except Exception as e:
        print(f"保存处理后的数据时发生错误: {e} (Error saving processed data: {e})")
        traceback.print_exc()


    return X_train, X_val, X_test, y_train, y_val, y_test, \
           label_encoders, \
           num_imputer if final_numerical_features else None, \
           scaler if final_numerical_features else None, \
           final_categorical_features, final_numerical_features


# --- 主执行示例 (Main Execution Example) ---
if __name__ == '__main__':
    try:
        from config import CSV_FILES, DEVELOPMENT_MODE, DEV_NROWS_PER_FILE
        
        print("🚀 启动SAINT数据预处理流程")
        print(f"📁 数据文件: {len(CSV_FILES)}个文件")
        
        if DEVELOPMENT_MODE:
            print(f"🔧 开发模式: 每文件限制{DEV_NROWS_PER_FILE}行")
            nrows_per_file = DEV_NROWS_PER_FILE
        else:
            print("🔥 生产模式: 处理全部数据")
            nrows_per_file = None
        
        print(f"开始主预处理流程 (Starting main preprocessing pipeline)")
        processed_outputs = main_preprocess_pipeline(CSV_FILES, nrows_per_file=nrows_per_file)
        
        X_train, X_val, X_test, y_train, y_val, y_test = processed_outputs[:6]

        print(f"\n✅ 预处理完成！数据统计:")
        print(f"   训练集: {X_train.shape}")
        print(f"   验证集: {X_val.shape}")
        print(f"   测试集: {X_test.shape}")
        print(f"   设备类别数: {len(y_train.unique())}")
        
        print(f"\n📊 类别分布 (前5类):")
        print(y_train.value_counts().head())

    except ImportError:
        print("⚠️  未找到config.py，使用默认配置")
        # 降级到硬编码路径
        filepath_list = [
            '/mnt/d/数据集/CIC/BenignTraffic.csv',
            '/mnt/d/数据集/CIC/BenignTraffic1.csv',
            '/mnt/d/数据集/CIC/BenignTraffic2.csv',
            '/mnt/d/数据集/CIC/BenignTraffic3.csv'
        ]
        processed_outputs = main_preprocess_pipeline(filepath_list, nrows_per_file=10000)  # 开发模式
        
    except FileNotFoundError:
        print(f"❌ 错误: 数据文件未找到。请检查config.py中的路径配置")
    except ValueError as ve:
        print(f"❌ 数据处理错误: {ve}")
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()