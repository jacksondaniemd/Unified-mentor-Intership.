import streamlit as st
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="K-Pop Momentum Dashboard", layout="wide")

st.title("🎵 K-Pop Comeback & Fandom Analytics")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Atlantic_South_Korea.csv")
    df.columns = df.columns.str.lower().str.strip()

    # Basic cleaning
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')

    df = df.dropna(subset=['date', 'position', 'popularity'])

    # Create ID
    df['song_artist'] = (
        df['song'].astype(str).str.lower().str.strip() + "_" +
        df['artist'].astype(str).str.lower().str.strip()
    )

    # Duration
    df['duration_min'] = df['duration_ms'] / 60000

    # Sort
    df = df.sort_values(['song_artist', 'date'])

    # Re-entry
    df['prev_date'] = df.groupby('song_artist')['date'].shift(1)
    df['gap_days'] = (df['date'] - df['prev_date']).dt.days

    df['entry_type'] = np.where(
        df['prev_date'].isna(), 'first_entry',
        np.where(df['gap_days'] > 1, 're_entry', 'continuous')
    )

    # Momentum
    df['prev_rank'] = df.groupby('song_artist')['position'].shift(1)
    df['rank_change'] = -(df['position'] - df['prev_rank'])

    df['prev_pop'] = df.groupby('song_artist')['popularity'].shift(1)
    df['pop_change'] = (df['popularity'] - df['prev_pop']) / df['prev_pop']
    df['pop_change'] = df['pop_change'].replace([np.inf, -np.inf], 0)

    df['momentum_score'] = df['rank_change'] * df['pop_change']

    # Recovery
    df['days_since_entry'] = df.groupby('song_artist').cumcount()
    df['recovery_speed'] = df['rank_change'] / (df['days_since_entry'] + 1)

    # Extra fields
    df['album_type'] = df.get('album_type', 'unknown')
    df['total_tracks'] = pd.to_numeric(df.get('total_tracks', 0), errors='coerce').fillna(0)
    df['is_explicit'] = df.get('is_explicit', False)
    df['is_explicit'] = df['is_explicit'].astype(str).str.lower().isin(['true','1'])

    return df

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

artists = st.sidebar.multiselect("Select Artist", df['artist'].dropna().unique())
album_type = st.sidebar.selectbox("Album Type", ["All"] + list(df['album_type'].unique()))
reentry_filter = st.sidebar.slider("Minimum Re-entries", 0, 10, 0)

filtered_df = df.copy()

if artists:
    filtered_df = filtered_df[filtered_df['artist'].isin(artists)]

if album_type != "All":
    filtered_df = filtered_df[filtered_df['album_type'] == album_type]

# Re-entry filter
reentry_counts = filtered_df.groupby('song_artist')['entry_type'].apply(lambda x: (x == 're_entry').sum())
valid_songs = reentry_counts[reentry_counts >= reentry_filter].index
filtered_df = filtered_df[filtered_df['song_artist'].isin(valid_songs)]

# =========================
# KPI SECTION
# =========================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Songs", filtered_df['song_artist'].nunique())
col2.metric("Avg Momentum", round(filtered_df['momentum_score'].mean(), 2))
col3.metric("Avg Recovery Speed", round(filtered_df['recovery_speed'].mean(), 2))
col4.metric("Re-entry Rate", round((filtered_df['entry_type'] == 're_entry').mean(), 2))

# =========================
# RE-ENTRY TIMELINE
# =========================
st.subheader("🔁 Re-entry Timeline")

timeline = filtered_df.groupby('date')['entry_type'].apply(lambda x: (x == 're_entry').sum())
st.line_chart(timeline)

# =========================
# MOMENTUM ANALYSIS
# =========================
st.subheader("⚡ Momentum Distribution")

st.scatter_chart(filtered_df[['rank_change', 'momentum_score']])

# =========================
# COMEBACK VS FIRST ENTRY
# =========================
st.subheader("📈 First Entry vs Re-entry")

compare = filtered_df.groupby('entry_type')['momentum_score'].mean()
st.bar_chart(compare)

# =========================
# CONTENT ANALYSIS
# =========================
st.subheader("🎧 Content Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Album Type vs Momentum")
    st.bar_chart(filtered_df.groupby('album_type')['momentum_score'].mean())

with col2:
    st.write("Explicit vs Clean")
    st.bar_chart(filtered_df.groupby('is_explicit')['momentum_score'].mean())

# =========================
# FANDOM LEADERBOARD
# =========================
st.subheader("🔥 Fandom Intensity Leaderboard")

summary = filtered_df.groupby('song_artist').agg(
    momentum=('momentum_score', 'mean'),
    recovery=('recovery_speed', 'mean'),
    reentries=('entry_type', lambda x: (x == 're_entry').sum())
).reset_index()

summary['fandom_score'] = (
    summary['reentries'] * summary['momentum']
) / (summary['recovery'] + 1)

top = summary.sort_values('fandom_score', ascending=False).head(10)

st.dataframe(top)

st.bar_chart(top.set_index('song_artist')['fandom_score'])

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built for K-Pop Momentum Analysis 🚀")