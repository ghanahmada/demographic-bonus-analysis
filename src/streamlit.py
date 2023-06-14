import json
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


st.set_page_config(page_title="Capstone Project", page_icon=":chart_with_upwards_trend:", layout="wide")

# Helper function
@st.cache_data()
def load_data(path):
    df = pd.read_csv(path)
    return df

def fig_geographic_map(locations, valuescale):
    with open("raw_data/data_map_indo/indonesia-prov.geojson") as response:
        geo = json.load(response)
    fig = go.Figure(go.Choroplethmapbox(geojson=geo, 
                                    locations=locations, 
                                    featureidkey="properties.Propinsi",
                                    z=valuescale,
                                    colorscale="YlOrRd", 
                                    marker_opacity=0.9,
                                    marker_line_width=0.3,
                                    colorbar=dict(
                                            tickmode="array",
                                            tickvals=[0, 1, 2, 3, 4], 
                                            ticktext=["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"], 
                                            tickfont=dict(color="grey"),
                                            len=0.6,
                                            y=0.5,
                                            x=1.02,)
                                    ))

    fig.update_layout(mapbox_style="carto-positron",
                            height = 425,
                            width = 800,
                            autosize=True,
                            margin={"r":0,"t":0,"l":0,"b":0},
                            mapbox=dict(center=dict(lat=-1.4393, lon=116.9213), zoom=3.4))
    return fig

def fig_linecorr(feature):
    min_max_scaler = lambda column: (column - column.min()) / (column.max() - column.min())
    fig_linecorr = go.Figure(data=[go.Scatter(x=data_corr["tahun"], name="Rasio Kemiskinan", line=dict(color="#F7BE6D"),
                                              y=min_max_scaler(data_corr[corr_target]), mode="lines"),
                                   go.Scatter(x=data_corr["tahun"], name=feature, line=dict(color="#F05E16"),
                                              y=min_max_scaler(data_corr[feature]), mode="lines")])
    fig_linecorr.update_layout(height=350, width=800, margin={"r":0,"t":0,"l":60,"b":0})
    return fig_linecorr

def sum_facilities(df: pd.DataFrame):
    df["total_fasped"] = (df["kelurahan_jumlah_sd"] + df["kelurahan_jumlah_smp"] + 
                          df["kelurahan_jumlah_sma"] + df["kelurahan_jumlah_smk"] + df["kelurahan_jumlah_pt"])
    df["total_faskes"] = (df["Jumlah Rumah Sakit Umum"] + df["Jumlah Rumah Sakit Khusus"] + 
                          df["Jumlah Puskesmas Rawat Inap"] + df["Jumlah Puskesmas Non Rawat Inap"])

def populate_graph():
   if st.session_state.year_range != st.session_state.previous_year_range:
      st.session_state.previous_year_range = st.session_state.year_range.copy()
      st.experimental_rerun()
   if st.session_state.plot_type != st.session_state.previous_plot_type:
      st.session_state.previous_plot_type = st.session_state.plot_type
      st.experimental_rerun()


# ===============================================================================

sun_colors = ['rgb(206, 75, 26, 0.7)', 'rgb(243, 190, 44, 0.7)', 'rgb(234, 135, 26, 0.7)']
colors = ['lightgoldenrodyellow',] * 5
colors[3] = 'orange'

# Plot Total Population Worldwide
population_df = (load_data(path="preprocessed_data/peringkat_total_penduduk_dunia.csv").iloc[:5, :]
                 .sort_values(by=["total_penduduk"], ascending=False))
fig_barplot = go.Figure(data=[go.Bar(y=population_df["negara"], x=population_df["total_penduduk"],
                                    marker=dict(color='rgba(170, 30, 20, 0.7)',
                                    line=dict(color='rgba(10, 30, 20, 1.0)',
                                    width=1)),
                                    orientation='h', marker_color=colors)])
fig_barplot.update_layout(yaxis=dict(autorange="reversed"), width=800,height=275,
                          margin={"r":0,"t":0,"l":0,"b":20})


# Plot Age Range
umur_df = load_data(path="preprocessed_data/data_klasifikasi_umur2.csv")
umur_df = umur_df[umur_df["tahun"].isin([2011, 2021])]
fig_age_bar = go.Figure(data=[
    go.Bar(name='Usia < 15 tahun', x=umur_df["tahun"], y=umur_df["usia_dibawah_15"], text=umur_df["usia_dibawah_15"], 
           marker_color="cornsilk", marker=dict(line=dict(color='rgba(10, 30, 20, 1.0)',
                                    width=1))),
    go.Bar(name='Usia 15-60 tahun', x=umur_df["tahun"], y=umur_df["usia_produktif"], text=umur_df["usia_produktif"], 
           marker_color="orange", marker=dict(line=dict(color='rgba(10, 30, 20, 1.0)',
                                    width=1))),
    go.Bar(name='Usia > 60 tahun', x=umur_df["tahun"], y=umur_df["usia_diatas_60"], text=umur_df["usia_diatas_60"], 
           marker_color="crimson", marker=dict(line=dict(color='rgba(10, 30, 20, 1.0)',
                                    width=1)))
])
fig_age_bar.update_layout(barmode='group', height=375, width=800, margin={"r":0,"t":0,"l":60,"b":0},
                          legend=dict(orientation="h", xanchor = "center", x = 0.3, y= 1))


# Plot Growth annual
def fig_growth_annual(start_year=2000, end_year=2021):
    growth_annual = load_data(path="preprocessed_data/data_pertumbuhan_penduduk.csv")
    growth_annual_res = growth_annual[(growth_annual["tahun"] >= start_year) & (growth_annual["tahun"] <= end_year)]

    def split_by_country(df):
        countries = ["China", "India", "United States", "Indonesia", "Pakistan"]
        return (df[df["negara"] == countries[i]] for i in range(len(countries)))

    growth_china, growth_india, growth_usa, growth_indo, growth_pks = split_by_country(growth_annual_res)

    fig_growth = go.Figure(data=[
        go.Scatter(name='Indonesia', x=growth_indo["tahun"], y=growth_indo["pertumbuhan_penduduk"], mode="lines"),
        go.Scatter(name='China', x=growth_china["tahun"], y=growth_china["pertumbuhan_penduduk"], mode="lines"),
        go.Scatter(name='India', x=growth_india["tahun"], y=growth_india["pertumbuhan_penduduk"], mode="lines"),
        go.Scatter(name='Pakistan', x=growth_pks["tahun"], y=growth_pks["pertumbuhan_penduduk"], mode="lines"),
        go.Scatter(name='United States', x=growth_usa["tahun"], y=growth_usa["pertumbuhan_penduduk"], mode="lines")
    ])

    fig_growth.update_layout(height=375, width=800, margin={"r":20,"t":0,"l":0,"b":20}, legend=dict(orientation="h", xanchor = "center", x = 0.3, y= 1))
    return fig_growth


# Plot Age Range
umur_df = load_data(path="preprocessed_data/data_klasifikasi_umur2.csv")
umur_df["tahun"] = umur_df["tahun"].astype(str)
used_cols = ["usia_dibawah_15", "usia_produktif", "usia_diatas_60"]

umur_18 = umur_df[umur_df["tahun"] == "2011"][used_cols].values[0]
umur_22 = umur_df[umur_df["tahun"] == "2021"][used_cols].values[0]

labels = ["Usia < 15 tahun", "Usia 15-60 tahun", "Usia > 60 tahun"]

fig_age = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig_age.add_trace(go.Pie(labels=labels, values=umur_18, marker_colors=sun_colors),
              1, 1)
fig_age.add_trace(go.Pie(labels=labels, values=umur_22),
              1, 2)

fig_age.update_traces(hole=.3, hoverinfo="label+percent+name")

fig_age.update_layout(height=375, width=600,margin={"r":0,"t":0,"l":60,"b":0},
    annotations=[dict(text='2011', x=0.19, y=0.49, font_size=15, showarrow=False),
                 dict(text='2021', x=0.81, y=0.49, font_size=15, showarrow=False)],
                 legend=dict(orientation="h", xanchor = "center", x = 0.3, y= 1))


# Plot Population Density
penduduk_prov = load_data(path="preprocessed_data/penduduk_per_provinsi_processed.csv")

fig_kepadatan = fig_geographic_map(locations=penduduk_prov["Provinsi"],
                                   valuescale=penduduk_prov["2022"])


# Visualization Clustering based on several aspects
cluster_df = load_data(path="preprocessed_data/cluster_data.csv")
sum_facilities(cluster_df)

X = cluster_df.drop(["Provinsi"], axis=1).copy()
quantity_cols = ["Jumlah Rumah Sakit Umum", "Jumlah Rumah Sakit Khusus", "Jumlah Puskesmas Rawat Inap", "Jumlah Puskesmas Non Rawat Inap",
                 "kelurahan_jumlah_sd", "kelurahan_jumlah_smp", "kelurahan_jumlah_sma", "kelurahan_jumlah_smk", "kelurahan_jumlah_pt"]

for col in quantity_cols:
    X[col] = X[col] / X["Luas Wilayah (km2)"]

sum_facilities(X)
X = X.drop(quantity_cols, axis=1)

mm_scaler = MinMaxScaler()
X_scaled = mm_scaler.fit_transform(X.values)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns).drop(["Luas Wilayah (km2)"], axis=1)

kmeanModel = KMeans(n_clusters=5, random_state=2023)
y=kmeanModel.fit(X_scaled).labels_
cluster_df_res = pd.concat([cluster_df, pd.DataFrame(y, columns=["no_cluster"])], axis=1)


fig_cluster = fig_geographic_map(locations=cluster_df_res["Provinsi"],
                                 valuescale=cluster_df_res["no_cluster"])


# Visualization heatmap correlation
data_nasional = load_data(path="preprocessed_data/data_nasional.csv")
data_corr = data_nasional[["tahun", "Populasi, total", "PDB (mata uang US$)", "Angka kematian, bayi (per 1.000 kelahiran hidup)", "Harapan hidup saat lahir, total (tahun)",
                           "Pengangguran, total (% dari angkatan kerja total)", "Indeks GINI", "Rasio jumlah masyarakat miskin pada garis kemiskinan nasional (% dari populasi)",
                           "Pertumbuhan PDB (% tahunan)", "Inflasi, deflator PDB (% tahunan)"]].dropna()
data_corr.rename(columns={"Populasi, total":"Total Populasi", "PDB (mata uang US$)": "PDB", "Angka kematian, bayi (per 1.000 kelahiran hidup)": "Angka kematian",
                          "Harapan hidup saat lahir, total (tahun)":"Harapan Hidup", "Pengangguran, total (% dari angkatan kerja total)":"Total Pengangguran",
                          "Indeks GINI":"Indeks GINI", "Rasio jumlah masyarakat miskin pada garis kemiskinan nasional (% dari populasi)": "Rasio Kemiskinan",
                          "Pertumbuhan PDB (% tahunan)":"Pertumbuhan PDB", "Inflasi, deflator PDB (% tahunan)": "Inflasi"}, inplace=True)

corr_target = "Rasio Kemiskinan"
fig_corr = plt.figure(figsize=(10,8))
new_df = {}
for i in np.setdiff1d(data_corr.drop(["tahun"], axis=1).columns.values, [corr_target]):
  new_df[i] = pearsonr(data_corr[corr_target],data_corr[i])
new_df = pd.DataFrame(new_df, index=[corr_target,'p_value']).T
sns.heatmap(new_df.sort_values(by=corr_target, ascending=False), annot=True, cmap="Oranges")


if __name__ == "__main__":
    if 'year_range' not in st.session_state: st.session_state.year_range = [2010, 2021]
    if 'previous_year_range' not in st.session_state: st.session_state.previous_year_range = [2010, 2021]
    if 'plot_type' not in st.session_state: st.session_state.plot_type = "Persentase"
    if 'previous_plot_type' not in st.session_state: st.session_state.previous_plot_type = "Persentase"

    JUDUL = st.title("Analisis Demografi dan Karakteristik Setiap Provinsi di Indonesia")
    st.divider()


    LEFT, MIDDLE, RIGHT = st.columns([1,1,1])
    LEFT.subheader("Latar Belakang")
    LEFT.markdown("""Menurut Badan Pusat Statistik (BPS), Indonesia diperkirakan akan mengalami masa **bonus demografi** 
                    pada periode **2020-2030**, di mana jumlah **penduduk usia produktif** akan melampaui jumlah **penduduk 
                    usia anak** dan **lanjut usia**. Keberadaan populasi usia produktif yang tinggi memiliki potensi besar sebagai 
                    sumber daya tenaga kerja yang dapat mempercepat **pembangunan ekonomi negara**. Namun, 
                    potensi bonus demografi ini juga dapat membawa risiko bencana demografi, yaitu tingginya **angka pengangguran** 
                    yang dapat mengakibatkan kenaikan **angka kemiskinan**.""")
    MIDDLE.subheader("  Jumlah Penduduk Terbanyak 2021")
    MIDDLE.plotly_chart(fig_barplot, use_container_width=True)
    RIGHT.subheader("Tujuan", anchor="right")
    RIGHT.markdown("1. Memberikan wawasan mengenai kondisi demografi di Indonesia saat ini")
    RIGHT.markdown("2. Menganalisis setiap wilayah di Indonesia berdasarkan faktor ekonomi, sosial, dan pembangunan")
    RIGHT.markdown("3. Mencari korelasi pertumbuhan penduduk dengan beberapa variabel demografi")


    LEFT2, RIGHT2 = st.columns(2)
    LEFT2.subheader("Persentase Pertumbuhan Penduduk")
    LEFT2.plotly_chart(fig_growth_annual(*st.session_state.year_range), use_container_width=True)
    RIGHT2.subheader("Perbandingan Komposisi Kelompok Usia di Indonesia")
    if st.session_state.plot_type == "Jumlah":
        RIGHT2.plotly_chart(fig_age_bar, use_container_width=True)
    else:
        RIGHT2.plotly_chart(fig_age, use_container_width=True)
    st.session_state.year_range[0], st.session_state.year_range[1] = LEFT2.slider("Pilih tahun mulai", min_value=2000, value=[2010,2021], max_value=2021)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.session_state.plot_type = RIGHT2.radio("",("Persentase","Jumlah"))


    st.subheader("Karakteristik Wilayah di Indonesia")
    LEFT3, RIGHT3 = st.columns([3,2])
    plot_map = LEFT3.selectbox("Plot Map", ("Clustering", "Persebaran penduduk"))
    if plot_map == "Persebaran penduduk":
        LEFT3.plotly_chart(fig_kepadatan, use_container_width=True)
        RIGHT3.subheader("Sekitar 55% dari total penduduk di Indonesia menempati di Pulau Jawa")
    elif plot_map == "Clustering":
        LEFT3.plotly_chart(fig_cluster, use_container_width=True)
        RIGHT3.subheader("  Deskripsi Cluster")
        RIGHT3.markdown("- **Cluster 0**: pertumbuhan ekonomi [**TINGGI-182K**], IPM [SEDANG-74], rasio kemiskinan [SEDANG-6.6%], fasilitas kesehatan dan pendidikan [SEDANG-2836]")
        RIGHT3.markdown("- **Cluster 1**: pertumbuhan ekonomi [SEDANG-61K], IPM [SEDANG-72], rasio kemiskinan [SEDANG-7.8%], fasilitas kesehatan dan pendidikan [SEDANG-3833]")
        RIGHT3.markdown("- **Cluster 2**: pertumbuhan ekonomi [**TINGGI SEKALI-293K**], IPM [**TINGGI-82**], rasio kemiskinan [RENDAH-4.6%], fasilitas kesehatan dan pendidikan [SEDANG-1597]")
        RIGHT3.markdown("- **Cluster 3**: pertumbuhan ekonomi [SEDANG-52K], IPM [SEDANG-72], rasio kemiskinan [SEDANG-9.7%], fasilitas kesehatan dan pendidikan [**TINGGI-17124**]")
        RIGHT3.markdown("- **Cluster 4**: pertumbuhan ekonomi [**RENDAH-41K**], IPM [RENDAH-68], rasio kemiskinan [**TINGGI-17.1%**], fasilitas kesehatan dan pendidikan [SEDANG-3095]")

    LEFT4, MIDDLE4, RIGHT4 = st.columns([4.25, 0.75, 5])
    LEFT4.subheader("Korelasi Pertumbuhan Penduduk")
    LEFT4.pyplot(fig_corr, use_container_width=True)
    with LEFT4:
        with st.expander("Grafik Garis Hubungan"):
            feature = st.selectbox("Pilih fitur", list(data_corr.drop(["tahun"], axis=1).columns))
            fig_line = fig_linecorr(feature)
            st.plotly_chart(fig_line, use_container_width=True)
            
    RIGHT4.subheader("Kesimpulan")
    RIGHT4.markdown("""- Pertumbuhan penduduk di Indonesia mengalami penurunan dan telah berada di bawah angka 1% sejak tahun 2018. 
                        Hal ini menunjukkan adanya **tren penurunan laju pertumbuhan penduduk** di negara ini.""")
    RIGHT4.markdown("""- Dalam rentang waktu 2011 hingga 2021, terjadi peningkatan jumlah penduduk usia 15-60 tahun di Indonesia sehingga 
                        mengindikasikan pergeseran demografi menuju **struktur penduduk yang lebih muda dan lebih berpotensi produktif**.""")
    RIGHT4.markdown("""- Pulau Jawa menjadi tempat tinggal bagi sekitar 55% dari total penduduk Indonesia. Hal ini menunjukkan 
                        konsentrasi penduduk yang tinggi di pulau ini sehingga dapat berdampak pada **ketimpangan ekonomi dan sosial antara provinsi di Indonesia**.""")

    st.divider()
    
    st.markdown('<div style="text-align: right;"><i>Ghana Ahmada Yudistira</i></div>', unsafe_allow_html=True)
    
    populate_graph()
