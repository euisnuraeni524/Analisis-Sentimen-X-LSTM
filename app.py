import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


from utils.preprocessing import run_preprocessing
from utils.labeling import load_lexicon_csv, label_dataframe, POL_MAP
from utils.modeling import train_lstm, plot_history, plot_confusion_matrix
from utils import modeling
from utils.pdf_report import generate_history_pdf
from collections import Counter
from datetime import datetime


st.set_page_config(page_title='Analisis Sentimen X — LSTM', layout='wide')


# --------------------- STATE ---------------------
def init_state():
    defaults = {
        'page': 'dashboard',
        'raw_df': None,
        'text_col': None,
        'date_col': None,
        'norm_kamus': None, # dict untuk normalisasi
        'proc_df': None,
        'lexicon': None,
        'labeled_df': None,
        'train_artifacts': None, # dict dari train_lstm
        'history': {
            'period_text': '-',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# -------------------- HELPERS NAVIGASI --------------------
def next_page(name: str):
    st.session_state.page = name

def back_to(name: str):
    # Kembali ke halaman tertentu TANPA reset session_state
    st.session_state.page = name

def reset_for_new_analysis():
    # Reset total state untuk mulai analisis baru
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    next_page('upload')


# Reset total untuk memulai analisis baru
def reset_for_new_analysis():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    next_page('upload')

# -------------------- HELPERS --------------------
def next_page(name):
    st.session_state.page = name

random.seed(42)

def resample_texts_labels(texts: list[str], labels: list[str], mode: str = 'max'):
    """
    Oversampling sederhana agar tiap kelas punya jumlah sampel sama.
    mode='max' -> semua kelas dibuat sebanyak kelas terbanyak.
    Return: texts_res, labels_res
    """
    idx_by_class = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(y, []).append(i)

    counts = {k: len(v) for k, v in idx_by_class.items()}
    if mode == 'max':
        target = max(counts.values())
    else:
        target = max(counts.values())

    new_idx = []
    for y, idxs in idx_by_class.items():
        if len(idxs) == 0:
            continue
        need = target - len(idxs)
        if need > 0:
            extra = random.choices(idxs, k=need)
            new_idx.extend(idxs + extra)
        else:
            new_idx.extend(idxs)

    random.shuffle(new_idx)
    texts_res = [texts[i] for i in new_idx]
    labels_res = [labels[i] for i in new_idx]
    return texts_res, labels_res

def pick_default_cols(df):
    text_guess = None
    date_guess = None
    for c in df.columns:
        lc = c.lower()
        if text_guess is None and lc in ['text','tweet','content','isi']:
            text_guess = c
        if date_guess is None and lc in ['date','tanggal','created_at','time','waktu']:
            date_guess = c
    # fallback
    if text_guess is None: text_guess = df.columns[0]
    return text_guess, date_guess


# ---------------------- UI ----------------------

def page_dashboard():
    st.title('Analisis Sentimen Pengguna Media Sosial X terhadap Program Makan Siang Gratis (LSTM)')
    st.markdown('Aplikasi skripsi end-to-end: **Upload → Preprocessing → Labeling (Lexicon) → Train LSTM → Evaluasi → History (PDF)**')


    if st.button('Mulai Analisis ▶️', use_container_width=True):
        next_page('upload')




def page_upload():
    st.header('Upload Data Mentah (CSV)')

    file = st.file_uploader('Unggah CSV berisi kolom teks & tanggal (opsional).', type=['csv'])

    # 👉 Jika belum ada file, jangan lanjut apa-apa
    if file is None:
        st.info('Silakan unggah file CSV untuk melanjutkan.')
        # (opsional) tombol kembali
        if st.button('⬅️ Kembali (Dashboard)'):
            back_to('dashboard')
        return

    # 👉 Coba baca CSV dengan beberapa fallback, tanpa bikin df sebelum sukses
    df = None
    errors = []

    # kandidat cara baca CSV yang sering berhasil di Windows/Excel
    read_trials = [
        {},  # default
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
        {"sep": ";"},                      # Excel ID sering pakai ';'
        {"sep": ";", "encoding": "latin-1"}
    ]

    for kw in read_trials:
        try:
            file.seek(0)  # penting: reset pointer sebelum coba lagi
            df = pd.read_csv(file, **kw)
            break
        except Exception as e:
            errors.append(f"{kw}: {e}")

    if df is None:
        st.error("Gagal membaca CSV. Coba simpan ulang file sebagai CSV standar (delimiter koma) atau kirim 5 baris pertama ke saya.")
        with st.expander("Detail error (teknis)"):
            st.write("\n\n".join(errors))
        return

    # ✅ Sampai sini df sudah aman
    st.session_state.raw_df = df
    st.success(f'Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom')

    # Pemetaan kolom
    def pick_default_cols(df):
        text_guess = None
        date_guess = None
        for c in df.columns:
            lc = c.lower()
            if text_guess is None and lc in ['text','tweet','content','isi']:
                text_guess = c
            if date_guess is None and lc in ['date','tanggal','created_at','time','waktu']:
                date_guess = c
        if text_guess is None:
            text_guess = df.columns[0]
        return text_guess, date_guess

    text_guess, date_guess = pick_default_cols(df)
    st.subheader('Pemetaan Kolom')
    cols = df.columns.tolist()
    st.session_state.text_col = st.selectbox('Kolom Teks', cols, index=cols.index(text_guess) if text_guess in cols else 0)

    date_options = ['<Tidak ada tanggal>'] + cols
    date_idx = 0 if date_guess is None else (date_options.index(date_guess) if date_guess in date_options else 0)
    date_choice = st.selectbox('Kolom Tanggal (opsional)', date_options, index=date_idx)
    st.session_state.date_col = None if date_choice == '<Tidak ada tanggal>' else date_choice

    st.subheader('Pratinjau Dataset Awal')
    st.dataframe(df.head(20), use_container_width=True)

    # Navigasi
    c1, c2 = st.columns(2)
    with c1:
        if st.button('⬅️ Kembali (Dashboard)'):
            back_to('dashboard')
    with c2:
        if st.button('Selanjutnya ➜ Preprocessing', type='primary'):
            next_page('preprocessing')

def page_preprocessing():
    st.header('Preprocessing Teks')
    raw = st.session_state.raw_df
    if raw is None:
        st.warning('Silakan upload data terlebih dahulu.'); return

    text_col = st.session_state.text_col
    date_col = st.session_state.date_col

    st.write('**Dataset Mentah**')
    st.dataframe(raw[[c for c in [text_col, date_col] if c in raw.columns]].head(20), use_container_width=True)


    with st.expander('(Opsional) Unggah Kamus Normalisasi (CSV: slang, baku)'):
        up_norm = st.file_uploader('Kamus normalisasi', type=['csv'], key='normcsv')
        custom_norm = None
        if up_norm is not None:
            nd = pd.read_csv(up_norm)
            nd.columns = [c.strip().lower() for c in nd.columns]
            # dukung 2 kolom: slang, baku
            if {'slang','baku'}.issubset(set(nd.columns)):
                custom_norm = dict(zip(nd['slang'].astype(str).str.lower().str.strip(), nd['baku'].astype(str).str.lower().str.strip()))
                st.info(f"Kamus normalisasi termuat: {len(custom_norm)} entri")
            else:
                st.error("CSV harus punya kolom 'slang' dan 'baku'")
        st.session_state.norm_kamus = custom_norm

    if st.button('Jalankan Preprocessing'):
        st.session_state.proc_df = run_preprocessing(raw, text_col, date_col, extra_norm=st.session_state.norm_kamus)
        st.success('Preprocessing selesai.')


    if st.session_state.proc_df is not None:
        st.subheader('Hasil Preprocessing (kolom tahap‑demi‑tahap)')
        show_cols = [
            text_col, '__raw_text__', 'clean_text', 'tokens', 'normalized', 'no_stop', 'stemmed_tokens', 'stemmed_text'
        ]
        if st.session_state.date_col: show_cols.append('__date__')
        st.dataframe(st.session_state.proc_df[show_cols].head(30), use_container_width=True)

        if st.button('⬅️ Kembali (Upload)'):
            back_to('upload')

        if st.button('Selanjutnya ➜ Labeling (Lexicon)', type='primary'):
            next_page('labeling')

def page_labeling():
    st.header('Labeling — Lexicon Based')
    dfp = st.session_state.proc_df
    if dfp is None:
        st.warning('Lakukan preprocessing terlebih dahulu.'); return

    up_lex = st.file_uploader('Lexicon CSV (word,polarity)', type=['csv'])

    if up_lex is not None:
        # pakai file yang di-upload
        try:
            lex = load_lexicon_csv(up_lex)
        except Exception as e:
            st.error(str(e)); return
    else:
        # coba cari file default; kalau tidak ada → gunakan embedded lexicon
        sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_lexicon_id.csv')
        if os.path.exists(sample_path):
            lex = load_lexicon_csv(sample_path)
            st.info(f"Memakai lexicon contoh dari file: {len(lex)} entri")
        else:
            # Embedded CSV sebagai fallback agar tidak FileNotFoundError
            default_csv = """word,polarity
bagus,1
baik,1
mantap,1
puas,1
mendukung,1
setuju,1
terbantu,1
legowo,1
tepat,1
bermanfaat,1
positif,1
buruk,-1
parah,-1
mengecewakan,-1
mahal,-1
bohong,-1
hoax,-1
korup,-1
gagal,-1
pencitraan,-1
buang,-1
negatif,-1
program,0
makan,0
siang,0
gratis,1
subsidi,0
bantuan,1
sekolah,0
anak,0
"""
            lex = load_lexicon_csv(io.StringIO(default_csv))
            st.info(f"Memakai lexicon embedded bawaan: {len(lex)} entri")

    st.session_state.lexicon = lex

    if st.button('Proses Labeling'):
        from utils.labeling import label_dataframe
        labeled = label_dataframe(dfp, 'stemmed_tokens', lex)
        st.session_state.labeled_df = labeled
        st.success('Labeling selesai.')

    if st.session_state.labeled_df is not None:
        st.subheader('Contoh Hasil Labeling')
        st.dataframe(
            st.session_state.labeled_df[[st.session_state.text_col, 'stemmed_text', 'lexicon_score', 'label']].head(30),
            use_container_width=True
        )
        if st.button('⬅️ Kembali (Preprocessing)'):
            back_to('preprocessing')

        if st.button('Selanjutnya ➜ Train Model LSTM', type='primary'):
            st.session_state.page = 'train'

def page_train():
    st.header('Training Model LSTM')
    dfl = st.session_state.labeled_df
    if dfl is None:
        st.warning('Selesaikan labeling terlebih dahulu.'); return

    # Tampilkan distribusi label saat ini
    vc = dfl['label'].value_counts()
    st.subheader('Distribusi Label (sebelum resampling)')
    st.write(vc.to_dict())

    # Validasi dasar sebelum training
    if vc.shape[0] < 2:
        st.error("Minimal harus ada ≥2 kelas berbeda agar model bisa dilatih.")
        return
    if (vc < 2).any():
        st.warning("Ada kelas dengan jumlah sangat sedikit (<2). Pertimbangkan memakai Resampling.")

    st.write('Pengaturan:')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_words = st.number_input('Vocab size (num_words)', min_value=2000, max_value=100000, value=20000, step=1000)
    with c2:
        max_len = st.number_input('Panjang sekuens (max_len)', min_value=32, max_value=512, value=100, step=8)
    with c3:
        epochs = st.number_input('Epochs', min_value=1, max_value=50, value=5, step=1)
    with c4:
        batch_size = st.number_input('Batch size', min_value=16, max_value=512, value=32, step=16)

    # Opsi resampling
    use_resampling = st.checkbox('Aktifkan Resampling (Oversampling acak per kelas)', value=False,
                                help='Jika dicentang, data akan di-oversample agar tiap kelas punya jumlah sampel sama (berdasarkan kelas terbanyak).')

    if st.button('Mulai Training ▶️'):
        texts = dfl['stemmed_text'].astype(str).tolist()
        labels = dfl['label'].astype(str).tolist()


        if use_resampling:
            texts, labels = resample_texts_labels(texts, labels, mode='max')
            # tampilkan distribusi setelah resampling
            from collections import Counter
            vc_after = Counter(labels)
            st.subheader('Distribusi Label (setelah resampling)')
            st.write(dict(vc_after))

        with st.spinner('Training berjalan...'):
            artifacts = train_lstm(texts, labels, num_words=num_words, max_len=max_len, epochs=epochs, batch_size=batch_size, model_dir='models')
            st.session_state.train_artifacts = artifacts
        st.success('Training selesai.')

    if st.session_state.train_artifacts is not None:
        st.subheader('Grafik Loss & Accuracy')
        h = st.session_state.train_artifacts['history']
        fig1, fig2 = plot_history(h)
        st.pyplot(fig1, use_container_width=False, clear_figure=True)
        st.pyplot(fig2, use_container_width=False, clear_figure=True)
        if st.button('⬅️ Kembali (Labeling)'):
            back_to('labeling')
        if st.button('Selanjutnya ➜ Evaluasi Model', type='primary'):
            next_page('evaluate')

def page_evaluate():
    st.header('Evaluasi Model')
    arts = st.session_state.train_artifacts
    if arts is None:
        st.warning('Latih model terlebih dahulu.')
        return

    cm = arts['metrics']['confusion_matrix']
    acc = arts['metrics']['accuracy']
    le = arts['label_encoder']

    st.metric('Accuracy (test set)', f"{acc:.4f}")
    fig_cm = plot_confusion_matrix(cm, le.classes_)
    st.pyplot(fig_cm, use_container_width=False, clear_figure=True)

    # === Tampilkan Classification Report (Precision/Recall/F1) ===
    st.subheader('Classification Report (Precision / Recall / F1)')

    rep = arts['metrics'].get('report')
    if rep is None:
        st.info('Classification report belum tersedia.')
        return

    rows = []
    for k, v in rep.items():
        if k == 'accuracy':
            continue
        rows.append({
            'Label': str(k),
            'Precision': v.get('precision', None),
            'Recall': v.get('recall', None),
            'F1': v.get('f1-score', None),
            'Support': v.get('support', None)
        })
    df_rep = pd.DataFrame(rows)

    base_labels = set(arts['label_encoder'].classes_.tolist())
    df_rep['__is_agg__'] = ~df_rep['Label'].isin(base_labels)
    df_rep = df_rep.sort_values(['__is_agg__', 'Label']).drop(columns='__is_agg__').reset_index(drop=True)

    st.dataframe(
        df_rep.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}', 'Support': '{:.0f}'}),
        use_container_width=True
    )

    if isinstance(rep.get('accuracy', None), (int, float)):
        st.metric('Overall Accuracy', f"{rep['accuracy']:.4f}")

    csv_bytes = df_rep.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Unduh Classification Report (CSV)',
        data=csv_bytes,
        file_name='classification_report.csv',
        mime='text/csv'
    )


    # simpan ke history
    st.session_state.history['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    if st.button('⬅️ Kembali (Training)'):
        back_to('train')

    if st.button('Selanjutnya ➜ History & PDF', type='primary'):
        next_page('history')

def page_history():
    st.header('History & Ekspor PDF')
    raw = st.session_state.raw_df
    dcol = st.session_state.date_col

    period_text = '-'
    if raw is not None and dcol and dcol in raw.columns:
        try:
            dts = pd.to_datetime(raw[dcol], errors='coerce')
            dmin, dmax = dts.min(), dts.max()
            if pd.notna(dmin) and pd.notna(dmax):
                period_text = f"{dmin.date()} s.d. {dmax.date()}"
        except Exception:
            pass

    st.session_state.history['period_text'] = period_text
    analysis_date = st.session_state.history['analysis_date']

    st.write('**Periode Data:**', period_text)
    st.write('**Tanggal Analisis:**', analysis_date)

    # tombol ekspor PDF
    labeled = st.session_state.labeled_df
    arts = st.session_state.train_artifacts
    acc = arts['metrics']['accuracy'] if arts else None
    label_counts = labeled['label'].value_counts().to_dict() if labeled is not None else {}

    # siapkan path (timestamp), tapi simpan yang berhasil ke session_state
    buf_path = os.path.join('models', f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    title = 'Laporan History Analisis Sentimen — Program Makan Siang Gratis'

    if st.button('Generate PDF'):
        os.makedirs('models', exist_ok=True)
        generate_history_pdf(
            output_path=buf_path,
            title=title,
            period_text=period_text,
            analysis_date=analysis_date,
            n_rows=(0 if labeled is None else len(labeled)),
            label_counts=label_counts,
            accuracy=acc,
            notes='Model LSTM dilatih dengan label lexicon (otomatis dari kamus).'
        )
        # simpan path terakhir agar bisa diunduh meski rerun
        st.session_state['last_history_pdf'] = buf_path
        st.success('PDF berhasil dibuat.')

    # tombol kembali (PERBAIKAN: tidak over-indent)
    if st.button('⬅️ Kembali (Evaluasi)'):
        back_to('evaluate')

    # unduh PDF terakhir bila ada
    last_pdf = st.session_state.get('last_history_pdf')
    if last_pdf and os.path.exists(last_pdf):
        with open(last_pdf, 'rb') as f:
            st.download_button(
                'Unduh PDF',
                data=f,
                file_name=os.path.basename(last_pdf),
                mime='application/pdf'
            )

    st.info('Selesai. Anda bisa kembali ke Dashboard atau mengulang proses dengan dataset berbeda.')

    # --- Tombol navigasi cepat ---
    c1, c2 = st.columns(2)
    with c1:
        if st.button('🏠 Kembali ke Dashboard'):
            next_page('dashboard')
    with c2:
        if st.button('🔄 Analisis Baru'):
            reset_for_new_analysis()

# -------------------- ROUTER --------------------
page_map = {
    'dashboard': page_dashboard,
    'upload': page_upload,
    'preprocessing': page_preprocessing,
    'labeling': page_labeling,
    'train': page_train,
    'evaluate': page_evaluate,
    'history': page_history,
}


page_map.get(st.session_state.page, page_dashboard)()