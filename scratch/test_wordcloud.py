import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df_valid = pd.read_csv('data/interim/valid_labeled_comments.csv')

aspect_names = {
    'infra_sentiment': 'Infrastruktur',
    'ekonomi_sentiment': 'Ekonomi & Harga',
    'kualitas_sentiment': 'Kualitas & Durabilitas',
    'purnajual_sentiment': 'Purnajual & Layanan'
}

sentiments = ['positif', 'netral', 'negatif']
sent_colors = {'positif': 'Greens', 'netral': 'YlOrBr', 'negatif': 'Reds'}

VISUALIZATION_STOPWORDS = set([
    'yang', 'yg', 'nya', 'di', 'ke', 'dan', 'ini', 'itu', 'ada', 'sudah', 'bisa', 'banyak', 
    'lagi', 'sama', 'kalau', 'kalo', 'akan', 'jadi', 'bikin', 'dari', 'pada', 'buat', 'saja', 
    'aja', 'atau', 'dengan', 'untuk', 'lah', 'pun', 'kan', 'kah', 'deh', 'dong', 'kok', 'juga', 
    'masih', 'belum', 'harus', 'gak', 'ga', 'ngga', 'nggak', 'tidak', 'tak', 'gk', 'apa', 'tapi', 
    'tetap', 'biar', 'pakai', 'pake', 'mau', 'orang', 'sih', 'lu', 'gue', 'gw', 
    'gua', 'dia', 'mereka', 'kita', 'kamu', 'anda', 'saya', 'aku', 'sy', 'om', 'bang', 'min', 
    'bro', 'bos', 'gan', 'sist', 'kak', 'bapak', 'ibu', 'pak', 'bu', 'terus', 
    'seperti', 'karena', 'sampai', 'jika', 'bila', 'semua', 'hal', 'bahkan', 'secara', 'malah', 
    'sebab', 'oleh', 'serta', 'tersebut', 'tentang', 'bahwa', 'apabila', 'kalau'
])

fig, axes = plt.subplots(4, 3, figsize=(18, 16))

for row_idx, (col, a_name) in enumerate(aspect_names.items()):
    for col_idx, sent in enumerate(sentiments):
        ax = axes[row_idx, col_idx]
        sub_df = df_valid[df_valid[col] == sent]
        texts = sub_df['text_cleaned'].dropna().astype(str).tolist()
        combined_text = " ".join(texts)
        
        if len(combined_text.strip()) > 0:
            wordcloud = WordCloud(
                width=500, height=350,
                background_color='white',
                colormap=sent_colors[sent],
                max_words=60,
                collocations=False,
                stopwords=VISUALIZATION_STOPWORDS
            ).generate(combined_text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f"{a_name} - {sent.capitalize()} (N={len(sub_df)})", fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "Data Kosong", ha='center', va='center', fontsize=12)
            ax.set_title(f"{a_name} - {sent.capitalize()} (N=0)", fontsize=11, fontweight='bold')
            
        ax.axis('off')

plt.suptitle("WordCloud Kata Kunci per Aspek & 3 Sentimen (Positif, Netral, Negatif)", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig("scratch/wordcloud_grid.png", bbox_inches='tight', dpi=150)
print("Saved wordcloud grid to scratch/wordcloud_grid.png")
