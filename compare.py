import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.set_page_config(page_title="Cross-Check Stok - TF-IDF Matching", layout="wide")
st.title("📦 Cross-Check Stok dengan TF-IDF Similarity")

# Upload files
col1, col2 = st.columns(2)
with col1:
    File_target = st.file_uploader("Upload Stok Target", type=['xlsx', 'xls'], key="target")
with col2:
    file_sumber = st.file_uploader("Upload Stok Sumber", type=['xlsx', 'xls'], key="sumber")

def process_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        df.columns = ['NAMA_BARANG', 'QTY']
        df = df.dropna(how='all')
        df['QTY'] = pd.to_numeric(df['QTY'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def tfidf_matching(list_sumber, list_target, threshold=0.5):
    """
    TF-IDF matching menggunakan cosine similarity
    
    Parameters:
    - list_sumber: list nama barang dari sumber
    - list_target: list nama barang dari target
    - threshold: minimum similarity score (0-1)
    
    Returns: DataFrame dengan hasil matching
    """
    
    # Combine semua nama untuk vectorization
    all_names = list_sumber + list_target
    
    # Create TF-IDF vectorizer
    # analyzer='char' untuk character-level matching (lebih fleksibel)
    # ngram_range=(2, 2) menggunakan bigram
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3),  # bigram dan trigram
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Fit dan transform semua nama
    tfidf_matrix = vectorizer.fit_transform(all_names)
    
    # Compute cosine similarity antara sumber dan target
    similarity_matrix = cosine_similarity(
        tfidf_matrix[:len(list_sumber)],
        tfidf_matrix[len(list_sumber):]
    )
    
    # Ekstrak best matches
    hasil_matching = []
    
    for i, sumber_name in enumerate(list_sumber):
        # Cari best match
        best_target_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_target_idx]
        
        if best_score >= threshold:
            best_target_name = list_target[best_target_idx]
            status = 'Match'
        else:
            best_target_name = None
            status = 'Tidak Ada'
        
        hasil_matching.append({
            'sumber_idx': i,
            'target_idx': best_target_idx if best_score >= threshold else None,
            'sumber_name': sumber_name,
            'target_name': best_target_name,
            'similarity_score': best_score,
            'status': status
        })
    
    return hasil_matching, similarity_matrix

if File_target is not None and file_sumber is not None:
    with st.spinner("Memproses data dengan TF-IDF..."):
        df_target = process_excel_file(File_target)
        df_sumber = process_excel_file(file_sumber)
        
        if df_target is not None and df_sumber is not None:
            st.success("✅ File berhasil diupload!")
            
            # Settings
            st.subheader("⚙️ Konfigurasi TF-IDF Matching")
            
            threshold = st.slider(
                "Threshold Similarity (0.0 - 1.0):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Semakin tinggi = semakin ketat. Rekomendasikan 0.4-0.6"
            )
            
            # Jalankan TF-IDF matching
            list_sumber = df_sumber['NAMA_BARANG'].tolist()
            list_target = df_target['NAMA_BARANG'].tolist()
            
            hasil_matching, similarity_matrix = tfidf_matching(
                list_sumber,
                list_target,
                threshold=threshold
            )
            
            # Build result dataframe
            df_hasil = pd.DataFrame(hasil_matching)
            
            # Add QTY from original dataframes
            df_hasil['qty_sumber'] = df_hasil['sumber_idx'].apply(
                lambda x: df_sumber.iloc[x]['QTY']
            )
            
            df_hasil['qty_target'] = df_hasil.apply(
                lambda row: df_target.iloc[row['target_idx']]['QTY'] 
                if row['target_idx'] is not None else None,
                axis=1
            )
            
            # Rename dan format kolom display
            df_display = df_hasil[[
                'sumber_name',
                'target_name',
                'qty_sumber',
                'qty_target',
                'similarity_score',
                'status'
            ]].copy()
            
            df_display.columns = [
                'Nama Barang sumber',
                'Nama target',
                'QTY sumber',
                'QTY target',
                'Score',
                'Status'
            ]
            
            df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.2f}")
            
            # Statistics
            st.subheader("📊 Statistik Matching")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Total Item sumber", len(df_sumber))
            with col_b:
                matched_count = len(df_hasil[df_hasil['status'] == 'Match'])
                st.metric("Item Cocok", matched_count)
            with col_c:
                not_matched = len(df_hasil[df_hasil['status'] == 'Tidak Ada'])
                st.metric("Item Tidak Cocok", not_matched)
            with col_d:
                avg_score = df_hasil['similarity_score'].mean()
                st.metric("Rata-rata Score", f"{avg_score:.2f}")
            
            # Tampilkan tabel hasil
            st.subheader("📋 Hasil Analisa Cross-Check Stok")
            st.dataframe(df_display, use_container_width=True, height=500)
            
            # Download button
            st.subheader("📥 Download Hasil")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Main Results
                df_display.to_excel(writer, sheet_name='Hasil Matching', index=False)
                
                # Sheet 2: Statistics
                df_stats = pd.DataFrame({
                    'Metrik': ['Total Item sumber', 'Item Cocok', 'Item Tidak Cocok', 'Akurasi %'],
                    'Nilai': [
                        len(df_sumber),
                        len(df_hasil[df_hasil['status'] == 'Match']),
                        len(df_hasil[df_hasil['status'] == 'Tidak Ada']),
                        f"{(len(df_hasil[df_hasil['status'] == 'Match']) / len(df_sumber) * 100):.1f}%"
                    ]
                })
                df_stats.to_excel(writer, sheet_name='Statistik', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name="tfidf_matching_stok.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            # Advanced Analysis
            st.subheader("🔍 Analisis Lanjutan")
            
            tab1, tab2, tab3 = st.tabs(["Filter & Sort", "Perbandingan Stok", "Quality Report"])
            
            with tab1:
                st.write("**Filter berdasarkan Status**")
                filter_status = st.selectbox(
                    "Pilih Status:",
                    ['Semua', 'Match', 'Tidak Ada'],
                    key='filter1'
                )
                
                if filter_status == 'Semua':
                    df_filtered = df_display
                else:
                    df_filtered = df_display[df_display['Status'] == filter_status]
                
                st.dataframe(df_filtered, use_container_width=True, height=400)
                st.metric("Jumlah Item", len(df_filtered))
            
            with tab2:
                st.write("**Perbandingan Stok (Item yang Match)**")
                df_compare = df_hasil[df_hasil['status'] == 'Match'].copy()
                
                if len(df_compare) > 0:
                    df_compare['selisih'] = df_compare['qty_sumber'] - df_compare['qty_target']
                    df_compare['status_stok'] = df_compare.apply(
                        lambda row: '🔴 target Lebih Rendah' if row['selisih'] > 0 
                        else '🟢 target Lebih Tinggi' if row['selisih'] < 0 
                        else '⚪ Sama',
                        axis=1
                    )
                    
                    df_compare_display = df_compare[[
                        'sumber_name',
                        'qty_sumber',
                        'qty_target',
                        'selisih',
                        'status_stok'
                    ]].sort_values('selisih', ascending=False)
                    
                    df_compare_display.columns = [
                        'Nama Barang',
                        'QTY sumber',
                        'QTY target',
                        'Selisih',
                        'Status'
                    ]
                    
                    st.dataframe(df_compare_display, use_container_width=True, height=400)
                    
                    # Summary stok
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        lower = len(df_compare[df_compare['selisih'] > 0])
                        st.metric("target Lebih Rendah", lower)
                    with col_y:
                        higher = len(df_compare[df_compare['selisih'] < 0])
                        st.metric("target Lebih Tinggi", higher)
                    with col_z:
                        same = len(df_compare[df_compare['selisih'] == 0])
                        st.metric("Stok Sama", same)
                else:
                    st.info("Tidak ada item yang cocok untuk analisis stok")
            
            with tab3:
                st.write("**Laporan Kualitas Matching**")
                
                df_quality = df_hasil.copy()
                df_quality['score_category'] = pd.cut(
                    df_quality['similarity_score'],
                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                    labels=['Rendah (0-0.3)', 'Sedang (0.3-0.5)', 'Tinggi (0.5-0.7)', 'Sangat Tinggi (0.7-1.0)']
                )
                
                score_dist = df_quality['score_category'].value_counts().sort_index()
                
                st.write("**Distribusi Quality Score:**")
                st.bar_chart(score_dist)
                
                st.write("**Detail Distribusi:**")
                for cat in ['Rendah (0-0.3)', 'Sedang (0.3-0.5)', 'Tinggi (0.5-0.7)', 'Sangat Tinggi (0.7-1.0)']:
                    count = len(df_quality[df_quality['score_category'] == cat])
                    pct = (count / len(df_quality) * 100) if len(df_quality) > 0 else 0
                    st.write(f"• {cat}: {count} items ({pct:.1f}%)")

else:
    st.info("👆 Upload kedua file untuk memulai analisis TF-IDF")
    
    with st.expander("ℹ️ Tentang TF-IDF Matching"):
        st.markdown("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)** adalah teknik machine learning yang:
        
        1. **Mengkonversi teks menjadi angka (vector)** menggunakan character-level ngrams
        2. **Menghitung similarity** antar vector menggunakan cosine similarity
        3. **Fleksibel** terhadap perbedaan format, spacing, dan typo
        4. **Powerful** untuk dataset dengan naming convention yang sangat berbeda
        
        **Keunggulan:**
        - ✅ Tidak perlu manual mapping
        - ✅ Bekerja baik dengan format yang sangat berbeda
        - ✅ Scientific dan reproducible
        - ✅ Bisa adjust threshold sesuai kebutuhan
        
        **Threshold Guide:**
        - 0.3-0.4: Sangat fleksibel (banyak false positives)
        - 0.5-0.6: Balanced (recommended)
        - 0.7-0.8: Ketat (mungkin miss beberapa match)
        - 0.8+: Sangat ketat (hanya exact-like matches)
        """)
