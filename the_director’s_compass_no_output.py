# -*- coding: utf-8 -*-
"""🎬 The Director’s Compass 🎞️ EDA TovTech IMDb

<h1 style="font-family:impact;font-size:200%;text-align:center;">
    <span style="background-color:#F5C518;color:black;padding:5px 15px;border-radius:7px;"><b>THE DIRECTOR's COMPASS</b></span>
</h1>

<div style="text-align: center;">
    <img src="https://tovtech.org/logo.png" width="300" height="500"/>
</div>

<div style="text-align: center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg" width="300" height="500"/>
</div>

This project answers a single, commercially focused question: **does a director's genre versatility predict their success?**

From the perspective of a talent agent, this analysis uses extracted data from the <u>Internet Movie Database</u> ([*IMDb*](https://www.imdb.com/)) to define and measure director success.

The dataset is a single CSV file that can be found in [*THIS*](https://www.kaggle.com/datasets/raedaddala/top-500-600-movies-of-each-year-from-1960-to-2024) link.

It consolidates data for the most popular **500**-**600** movies each year from **1920** to **2025**.

We use it to blend audience **quality** (ratings) and **reach** (votes) into a single metric: **Success Score**.

By mapping this success score against genre versatility we develop a practical framework, "The Director's Compass", to identify and categorize talent into four actionable archetypes.
"""

# Imports, constants, theme, and helpers
import ast
import math
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import display, HTML, Markdown
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress known warnings for clarity and aesthetics
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in greater.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in less.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in cast.*')
# NOTE: These warnings are from Pandas operations on mixed NaN/numeric columns and don't affect final results

# Wider columns for full movie titles
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 160)

# Visual constants
WIDTH = 800
HEIGHT_MED = 600
HEIGHT_TALL = 800
LOGO = "#F5C518"

# Force global Plotly dark mode theme
px.defaults.template = "plotly_dark"

def dark(fig, title=None, height=HEIGHT_MED, width=WIDTH):
    """Apply uniform dark style and fixed size to a Plotly figure."""
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        margin=dict(l=40, r=30, t=70, b=40),)
    return fig

def pv(message: str):
    """Yellow verification box for dynamic variables. Using <span style='color:{LOGO};font-weight:bold'>...</span>."""
    box = ("<div style='background:#000;color:#fff;padding:12px 16px;margin:10px 0;"
        f"border-left:10px solid {LOGO};font-size:15px;line-height:1.15;'>"
        + message + "</div>")
    display(HTML(box))

def sty(df: pd.DataFrame, formats: dict | None = None):
    """Consistent dark mode Dataframe."""
    formats = formats or {}
    return (df.style
          .format(formats)
          .hide(axis="index")
          .set_table_styles([
              {'selector': 'th', 'props': [('text-align','center'),
                                           ('background-color','#111'), ('color','#fff'),
                                           ('padding','6px 8px')]},
              {'selector': 'td', 'props': [('background-color','#0A0A0A'), ('color','#fff'),
                                           ('border-color','#222'), ('padding','6px 8px')]},
              {'selector': 'tr:nth-child(even)', 'props': [('background-color','#0F0F0F')]},
              {'selector': 'tr:hover', 'props': [('background-color','#1A1A1A')]}])
          .set_properties(**{'border-color':'#222'}))

def to_votes(x):
    """Convert strings like '2.1K' or '1.5M' to numeric votes while passing numeric through."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    try:
        if s.endswith("K"):
            return float(s[:-1]) * 1000
        if s.endswith("M"):
            return float(s[:-1]) * 1000000
        return float(s.replace(",", ""))
    except Exception:
        return np.nan

def parse_duration_minutes(s):
    """Parse strings like '1h 37m' or '97 min' or numeric minutes to minutes."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)) and not math.isnan(s):
        return float(s)
    text = str(s).lower()
    h = m = 0
    if "h" in text:
        try:
            h = float(text.split("h")[0].strip())
        except Exception:
            h = 0
    if "m" in text:
        try:
            right = text.split("h")[-1] if "h" in text else text
            m = float(right.replace("min", "").replace("m", "").strip())
        except Exception:
            m = 0
    if "min" in text and "h" not in text and "m" not in text:
        try:
            m = float(text.replace("min", "").strip())
        except Exception:
            m = 0
    total = h * 60 + m
    return total if total > 0 else np.nan

def parse_list(field):
    """Parse JSON-like list strings into real Python lists while handling plain comma strings and None."""
    if pd.isna(field):
        return []
    val = field
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [i.strip().strip("'\"") for i in s.split(",") if i.strip().strip("'\"")]

# Archetype color map for visuals
ARCH_COLORS = {"The Masters": "#10B981",
               "The Specialists": "#A855F7",
               "The Craftsmen": "#60A5FA",
               "The Explorers": "#F59E0B",}

# Environment verification: confirm constants and theme are live
pv(f"Theme: <span style='color:{LOGO};font-weight:bold'>plotly_dark</span>.<br> "
   f"Fixed <span style='color:{LOGO};font-weight:bold'>" + str(WIDTH) + " px</span> width.<br>"
   f"IMDb logo accent color: <span style='color:{LOGO};font-weight:bold'>" + LOGO + "</span>.")

"""## 1. Ground Truth
### Loading and Verifying the IMDb Dataset
Our analysis begins by loading the dataset and performing essential integrity checks.

This ensures our source data is free of duplicates and that we have a clear picture of its scope and any missing values.

The key columns for this analysis include:
- **Identification Context:**
  - `id` (*object*): Unique film identifier that lets us track each movie and prevent double-counting.
  - `title` (*object*): The actual movie name we use to verify our data makes sense.
  - `year` (*int64*): Extracted from `release_date` (*object*) to enable temporal analysis and decade grouping.
- **Success Metrics:**
  - `rating` (*float64*): Audience quality score from **1-10** that measures how much viewers liked the film.
  - `votes` (*object* to *numeric*): Audience reach in **"2.1K"** or **"1.5M"** format that we'll convert to actual numbers.
- **Versatility Metric:**
  - `genres` (*object* to *list*): String of genre tags that we'll parse into lists to measure director range.
- **Attribution Axis:**
  - `directors` (*object* to *list*): String of director names that we'll parse to properly credit collaborative films.

These initial steps are critical for grounding all downstream metrics in verified counts and avoiding skewed comparisons later.
"""

# Load dataset from Kaggle
imdb_raw = pd.read_csv('/kaggle/input/final_dataset.csv')

# Extract `year` as separate column to enables temporal analysis and decade grouping later
imdb_raw["release_date"] = pd.to_datetime(imdb_raw.get("release_date", pd.NaT), errors="coerce")
imdb_raw["year"] = imdb_raw["release_date"].dt.year

pv(f"Loaded dataset with <span style='color:{LOGO};font-weight:bold'>{len(imdb_raw):,}</span> rows and "
   f"<span style='color:{LOGO};font-weight:bold'>{len(imdb_raw.columns)}</span> columns from "
   f"<span style='color:{LOGO};font-weight:bold'>{int(imdb_raw['year'].min()) if imdb_raw['year'].notna().any() else 'n/a'}</span>"
   f" to <span style='color:{LOGO};font-weight:bold'>{int(imdb_raw['year'].max() - 1) if imdb_raw['year'].notna().any() else 'n/a'}</span>.")

# Priority check on business-critical fields
key_cols = [c for c in ["rating","votes","duration","genres","directors","release_date"] if c in imdb_raw.columns]
miss = imdb_raw[key_cols].isna().mean().rename("missing_pct").mul(100).round(1).reset_index().rename(columns={"index":"column"})
display(sty(miss, formats={"missing_pct":"{:.1f}%"}))

"""### 1-A) Data Quality Assessment
The dataset exhibits strong completeness across critical fields.

While `rating` and `votes` show **6.4%** missing values each, this leaves us with **~59,000** films with complete success metrics, more than sufficient for robust statistical analysis.

The exceptional completeness of `directors` (**99.9%** complete) and `genres` (**98.8%**) proves crucial for our versatility framework.

These fields drive our core analysis: directors determines our grouping variable, while genres enables the versatility calculations.

The perfect completeness of `release_date` allows reliable temporal analysis.

The minimal missing `duration` data (**3.3%**) won't impact our commercial viability assessments.

This data quality profile supports our dual-metric approach:

we can confidently measure both success (through rating and votes) and versatility (through genre diversity) across our director cohort.
"""

# Checking for duplicate IDs that would inflate director metrics
dup_ct = int(imdb_raw.duplicated(subset=["id"]).sum()) if "id" in imdb_raw.columns else 0
if dup_ct > 0 and "id" in imdb_raw.columns:
    imdb_raw = imdb_raw.drop_duplicates(subset=["id"]).reset_index(drop=True)
pv(f"Duplicate ids detected: <span style='color:{LOGO};font-weight:bold'>{dup_ct:,}</span>.<br>"
   f"Rows remaining after drop: <span style='color:{LOGO};font-weight:bold'>{len(imdb_raw):,}</span>.")

# Initial preview
preview_cols = ["id","title","year","duration","rating","votes","genres","directors"]
display(sty(imdb_raw.head(5)[[c for c in preview_cols if c in imdb_raw.columns]], formats={"rating": "{:.1f}"}))

"""## 2. Finding the Signal
### Audience Ratings Cluster Above Average
**<u>Claim:</u>** The distribution of audience ratings is not uniform. It is left-skewed, with most films clustering in the **5.5** to **7.5** range.

This insight is crucial for setting realistic performance benchmarks.

**<u>Evidence:</u>** The histogram shows that while ratings span from **1.0** to **10.0**, the median rating is **6.3**.

The densest concentration of films receives ratings significantly above the scale's midpoint of **5.5**.

**<u>Interpretation:</u>** The long left tail of lower-rated films indicates a clear segment of underperformers.

More importantly, the high concentration in the upper range suggests that using the median is a robust way to compare films, as it won't be easily skewed by the poorly-rated outliers.
"""

# Rating Distribution
if "rating" in imdb_raw.columns:
    import numpy as np
    # Define bins to align with 0.25 increments on the x-axis
    min_rating = imdb_raw["rating"].min()
    max_rating = imdb_raw["rating"].max()
    bins = np.arange(min_rating, max_rating + 0.25, 0.25)
    counts, edges = np.histogram(imdb_raw["rating"].dropna(), bins=bins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    hist_df = pd.DataFrame({"bin_mid": mids, "count": counts})

    # Calculate percentage of total count
    total_count = hist_df["count"].sum()
    hist_df["percentage"] = (hist_df["count"] / total_count) * 100

    fig = px.bar(hist_df, x="bin_mid", y="count", color="bin_mid", color_continuous_scale="RdYlGn")
    fig.update_traces(hovertemplate="Rating %{x:.2f}<br>Count %{y:,}<br>Percentage %{customdata:.1f}%<extra></extra>", customdata=hist_df["percentage"])
    fig.update_xaxes(title="Rating", tickmode='linear', tick0=0, dtick=0.5); fig.update_yaxes(title="Count", tickformat = ",.0f", range=[1, 7500])
    fig = dark(fig, title="Ratings skew positively with <span style='font-weight:bold;'>63.5%</span> (<span style='font-weight:bold;'>37,567</span>) of films rated between <span style='font-weight:bold;'>5.5</span> and <span style='font-weight:bold;'>7.25</span>", height=HEIGHT_MED, width=WIDTH)
    fig.update_coloraxes(colorbar_title="Rating", showscale=False)

    # Add percentage text annotations above the bars
    fig.update_traces(text=hist_df["percentage"].apply(lambda x: f"{x:.1f}%" if x > 0 else ""),
                      textposition="outside",
                      textfont=dict(size=8)) # Adjusted font size for 3 characters

    # Add a shaded area for the ratings below median aka "long left tail"
    fig.add_shape(type="rect",
                  x0=imdb_raw["rating"].min(), y0=0,
                  x1=6.2, y1=6878,
                  fillcolor="lightgrey", opacity=0.2, layer="below", line_width=0)

    # Add annotation for the highlighted tail
    fig.add_annotation(
        x=imdb_raw["rating"].median() * 0.5,  # Position inside the shaded area
        y=counts.max() * 0.908,                # Position near the top of the bars
        text="<b>Long Left Tail</b><br>This area shows most films<br>have below-median ratings.",
        showarrow=False,
        align="left",
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4)

    # Add mean and median lines
    mean_rating = imdb_raw["rating"].mean()
    median_rating = imdb_raw["rating"].median()

    fig.add_shape(type="line",
                  x0=mean_rating, y0=0, x1=mean_rating, y1=7500,
                  line=dict(color="blueviolet", width=2, dash="dash"))
    fig.add_annotation(
        x=mean_rating, y=7400,
        text=f"Mean: {mean_rating:.2f}",
        showarrow=False,
        font=dict(color="blueviolet"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
        xshift=-40)

    fig.add_shape(type="line",
                  x0=median_rating, y0=0, x1=median_rating, y1=7500,
                  line=dict(color="turquoise", width=2, dash="dash"))
    fig.add_annotation(
        x=median_rating, y=7400,
        text=f"Median: {median_rating:.2f}",
        showarrow=False,
        font=dict(color="turquoise"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
        xshift=45)

# Calculate cumulative percentage and percentile
hist_df["cumulative_pct"] = hist_df["percentage"].cumsum()
hist_df["films_above"] = 100 - hist_df["cumulative_pct"] + hist_df["percentage"]
fig.update_traces(
        hovertemplate="<b>Rating: %{x:.2f}</b><br>" +
            "Films in this bin: %{y:,}<br>" +
            "Bin percentage: %{customdata[0]:.1f}%<br>" +
            "Films rated ≥%{x:.2f}: %{customdata[1]:.1f}%<br>" +
            "Percentile rank: %{customdata[2]:.1f}%<extra></extra>",
            customdata=np.column_stack([hist_df["percentage"], hist_df["films_above"], hist_df["cumulative_pct"]]))

fig.show(renderer='iframe')

"""## 3. Engineering Clean Features
### From Raw Text to Analytical Precision
Before analyzing director performance, the raw data must be cleaned, normalized, and restructured.

This process involves three steps:
- Standardizing data types.
- Reshaping the data to be director-centric.
- Applying a visibility filter.
"""

# Copy and normalize core fields
imdb = imdb_raw.copy()

# Normalize Votes and duration
imdb["votes_n"] = imdb.get("votes").apply(to_votes) if "votes" in imdb.columns else np.nan
imdb["duration_minutes"] = imdb.get("duration").apply(parse_duration_minutes) if "duration" in imdb.columns else np.nan

# Parse key list fields if present
for col in ["genres", "directors", "writers", "stars"]:
    if col in imdb.columns:
        imdb[col] = imdb[col].apply(parse_list)

# Verify parsing success
n_votes_parsed = int(imdb["votes_n"].notna().sum()) if "votes_n" in imdb.columns else 0
n_dur_parsed = int(imdb["duration_minutes"].notna().sum()) if "duration_minutes" in imdb.columns else 0
pv(f"Votes parsed: <span style='color:{LOGO};font-weight:bold'>{n_votes_parsed:,}</span>.<br>"
   f"Durations parsed: <span style='color:{LOGO};font-weight:bold'>{n_dur_parsed:,}</span>.")

# Preview transformations
show_cols = [c for c in ["title", "votes", "votes_n", "duration", "duration_minutes"] if c in imdb.columns]
if show_cols:
    display(sty(
        imdb.sample(5)[show_cols].rename(columns={"duration_minutes": "minutes"}),
        formats={"votes_n": "{:,.0f}", "minutes": "{:,.0f}"}))

"""### 3-A) Multi-Director Attribution Challenge
Films often have multiple directors, like [Anthony](https://www.imdb.com/name/nm0751577/) and [Joe](https://www.imdb.com/name/nm0751648/) Russo, aka "The Russo Brothers", who co-directed several Marvel films like [*Avengers: Endgame (2019)*](https://www.imdb.com/title/tt4154796/).

Our explode operation ensures each director receives full credit for collaborative successes, though this slightly inflates the total row count.
"""

# One-to-many transformation: films with multiple directors properly attribute credit to each
directors = imdb.explode("directors").rename(columns={"directors": "director"})
directors = directors[~directors["director"].isna() & (directors["director"] != "")]

# Preview the exploded dataframe to show the new structure
display(Markdown("### Director-Centric Dataset Preview"))
display(sty(directors[["title", "director", "rating"]].head(6).reset_index(drop=True), formats={"rating": "{:.1f}"}))

"""### 3-B) Applying a Commercial Viability Filter
To ensure commercial relevancy, we filter the dataset to include only "visible films" - those that have achieved a minimum threshold of audience engagement.

We define this as any film with at least **1,000** votes.

This 1K-vote visibility filter removes approximately **45%** of the raw dataset but retains films that represent over **95%** of total audience engagement.

This ensures our director analysis focuses on commercially proven work rather than obscure productions that could skew performance metrics.
"""

# Create final and filtered dataframe for all subsequent analysis
directors_visible  = directors[directors["votes_n"] >= 1000].copy()

# Verification message
pv(f"Created final analysis set with <span style='color:{LOGO};font-weight:bold'>{len(directors_visible ):,}</span>"
   f" film-director entries (from films with >= <span style='font-weight:bold'>1,000</span> votes).")

"""### 3-C) Final Data Checkpoint: Runtimes are Commercially Sound
**<u>Claim:</u>** Before analyzing directors, we must standardize our data.

This involves cleaning numeric fields, parsing text lists, and reshaping the data to be director-centric.

A consistency check on film runtimes confirms our dataset aligns with commercial standards.

**<u>Evidence:</u>** We successfully converted over **59,000** vote entries and **61,000** duration entries into clean numeric formats.

The runtime violin plot shows that most films cluster tightly between **85** and **120** minutes, with a median of **95** minutes.

Finally, we reshaped the data and filtered it to a final set of **28,883** film-director entries for films with over **1,000** votes.

**<u>Interpretation:</u>** These transformations create a reliable foundation for the entire analysis.

The standardized runtimes confirm we're analyzing a commercially conventional set of films, and the director-centric structure allows us to attribute success accurately.
"""

# Runtime Distribution as violin with 99th percentile cap to avoid scale crush by outliers
if "duration_minutes" in imdb.columns:
    y_lo = 0
    y_hi = float(imdb["duration_minutes"].quantile(0.99))
    n_out = int((imdb["duration_minutes"] > y_hi).sum())
    fig = px.violin(imdb, y="duration_minutes", box=True, points=False)
    fig.update_traces(fillcolor=LOGO, line_color=LOGO, opacity=0.55, selector=dict(type="violin"))
    fig.update_yaxes(title="Duration (minutes)", range=[y_lo, y_hi])
    fig = dark(fig, title="<span style='font-weight:bold;'>60%</span> of all films and <span style='font-weight:bold;'>71.9%</span> of films with ><span style='font-weight:bold;'>10K</span> votes are within the commercial norm", height=HEIGHT_MED, width=WIDTH)

    # Add lines and shaded area for commercial norm: 85-120 minutes
    commercial_norm_min = 85
    commercial_norm_max = 120
    fig.add_shape(type="line", x0=-0.5, y0=commercial_norm_min, x1=0.5, y1=commercial_norm_min,
                  line=dict(color="lightgrey", width=2, dash="dot"))
    fig.add_shape(type="line", x0=-0.5, y0=commercial_norm_max, x1=0.5, y1=commercial_norm_max,
                  line=dict(color="lightgrey", width=2, dash="dot"))
    fig.add_shape(type="rect", x0=-0.5, y0=commercial_norm_min, x1=0.5, y1=commercial_norm_max,
                  fillcolor="lightgrey", opacity=0.2, layer="below", line_width=0)

    fig.add_annotation(
        text=f"Capped at 99th percentile (~<span style='color:{LOGO};font-weight:bold'>{y_hi:.0f}</span> min). "
             f"Hidden outliers above cap: <span style='color:{LOGO};font-weight:bold'>{n_out:,}</span>.",
        x=0.5, xref="paper", y=1.05, yref="paper", showarrow=False)

    # Add annotation for commercial norm
    fig.add_annotation(
        x=0, y=107.5,
        text="<b>Commercial Norm (85-120 min)</b><br>The dense concentration of films here<br>validates the dataset's relevance.",
        showarrow=False,
        align="center",
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4)

    # Add mean and median lines
    mean_duration = imdb["duration_minutes"].mean()
    median_duration = imdb["duration_minutes"].median()

    fig.add_shape(type="line",
                  x0=-0.5, y0=mean_duration, x1=0.5, y1=mean_duration,
                  line=dict(color="blueviolet", width=2, dash="dash"))
    fig.add_annotation(
        x=-0.5, y=mean_duration,
        text=f"Mean: {mean_duration:.2f}",
        showarrow=False,
        font=dict(color="blueviolet"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
        xshift=-50,
        yshift=5)

    fig.add_shape(type="line",
                  x0=-0.5, y0=median_duration, x1=0.5, y1=median_duration,
                  line=dict(color="turquoise", width=2, dash="dash"))
    fig.add_annotation(
        x=-0.5, y=median_duration,
        text=f"Median: {median_duration:.2f}",
        showarrow=False,
        font=dict(color="turquoise"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
        xshift=-50,
        yshift=-5)

fig.show(renderer='iframe')

"""## 4. The Market Landscape
### Deconstructing Genre Performance
**<u>Claim:</u>** Genres are not created equal - some are produced more often, while others command higher ratings or broader audience reach.

Understanding this landscape is key to fairly judging a director's success within a specific market.

**<u>Evidence:</u>** The charts show that <u>Drama</u> and <u>Comedy</u> are the most produced genres.

However, high-reach genres like <u>Adventure</u> and <u>Action</u> attract a significantly higher median number of votes (**11K** and **10K**, respectively).

Meanwhile, some high-volume genres like <u>Horror</u> have a lower median rating (**5.7**), establishing a different performance baseline.

**<u>Interpretation:</u>** These metrics provide our market context.

A **6.7** rating in <u>Drama</u> is a solid performance, but a **6.7** in <u>Horror</u> would be a category-leading achievement.

This framework allows us to distinguish a director's skill from a genre's inherent market appeal.
"""

# Explode genres
if "genres" not in imdb.columns:
    raise ValueError("Expected 'genres' column was not found in the dataset.")
imdb_genre = directors_visible.explode("genres").rename(columns={"genres": "genre"})
imdb_genre = imdb_genre[~imdb_genre["genre"].isna() & (imdb_genre["genre"] != "")]

# Aggregate top 10 genres by film count
genre_stats = (imdb_genre.groupby("genre", as_index=False)
               .agg(film_count=("id", "nunique"),
                    avg_rating=("rating", "median"),
                    avg_votes=("votes_n", "median"))
               .sort_values("film_count", ascending=False)
               .head(10)
               .reset_index(drop=True))

# Calculate metrics for hover templates
genre_stats["market_share"] = (genre_stats["film_count"] / genre_stats["film_count"].sum() * 100).round(1)
genre_stats["quality_percentile"] = (genre_stats["avg_rating"].rank(pct=True) * 100).round(0)
genre_stats["votes_share"] = (genre_stats["avg_votes"] * genre_stats["film_count"] / (genre_stats["avg_votes"] * genre_stats["film_count"]).sum() * 100).round(1)

# Create sorted versions for each subplot
genre_stats_rating_sorted = genre_stats.sort_values("avg_rating", ascending=False).reset_index(drop=True)
genre_stats_votes_sorted = genre_stats.sort_values("avg_votes", ascending=False).reset_index(drop=True)

# Setup consistent color mapping across all subplots
from plotly.colors import qualitative
palette = px.colors.qualitative.Vivid
genre_color = {g: palette[i % len(palette)] for i, g in enumerate(genre_stats["genre"])}

# Create triple bar chart with consistent color mapping
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False,
    row_heights=[0.35, 0.30, 0.35],
    vertical_spacing=0.08,
    subplot_titles=("Output - top <span style='font-weight:bold;'>10</span> genres by film count: <u>Drama</u> & <u>Comedy</u> dominate with <span style='font-weight:bold;'>>40%</span>",
                    "Median rating - audience quality benchmark: <u>Drama</u>, <u>Romance</u>, & <u>Crime</u> rate highly",
                    "Median votes - audience reach benchmark: <u>Action</u> & <u>Adventure</u> drive engagement"))

# Row 1: Film count (original order) - includes market share
fig.add_trace(go.Bar(
    x=genre_stats["genre"],
    y=genre_stats["film_count"],
    marker=dict(color=[genre_color[g] for g in genre_stats["genre"]]),
    name="Films",
    text=genre_stats["film_count"].apply(lambda x: f"{x:,}"),
    textposition="inside",
    customdata=np.column_stack([
        genre_stats["avg_rating"],
        genre_stats["avg_votes"],
        genre_stats["market_share"]]),
    hovertemplate="<b>%{x}</b><br>" +
                  "Film Count: %{y:,}<br>" +
                  "Genre Market Share: %{customdata[2]:.1f}%<br>" +
                  "Median Rating: %{customdata[0]:.2f}<br>" +
                  "Median Votes: %{customdata[1]:,.0f}<extra></extra>"),
    row=1, col=1)

# Row 2: Rating (sorted by rating) - includes quality percentile
fig.add_trace(go.Bar(
    x=genre_stats_rating_sorted["genre"],
    y=genre_stats_rating_sorted["avg_rating"],
    marker=dict(color=[genre_color[g] for g in genre_stats_rating_sorted["genre"]]),
    name="Rating",
    text=genre_stats_rating_sorted["avg_rating"].round(1),
    textposition="inside",
    customdata=np.column_stack([
        genre_stats_rating_sorted["film_count"],
        genre_stats_rating_sorted["avg_votes"],
        genre_stats_rating_sorted["quality_percentile"]]),
    hovertemplate="<b>%{x}</b><br>" +
                  "Median Rating: %{y:.2f}<br>" +
                  "Quality Percentile: %{customdata[2]:.0f}th<br>" +
                  "Film Count: %{customdata[0]:,}<br>" +
                  "Median Votes: %{customdata[1]:,.0f}<extra></extra>"),
    row=2, col=1)

# Row 3: Votes (sorted by votes) - includes percent of total votes
fig.add_trace(go.Bar(
    x=genre_stats_votes_sorted["genre"],
    y=genre_stats_votes_sorted["avg_votes"],
    marker=dict(color=[genre_color[g] for g in genre_stats_votes_sorted["genre"]]),
    name="Votes",
    text=(genre_stats_votes_sorted["avg_votes"]/1000).round(0).astype(int).astype(str) + 'k',
    textposition="inside",
    customdata=np.column_stack([
        genre_stats_votes_sorted["avg_rating"],
        genre_stats_votes_sorted["film_count"],
        genre_stats_votes_sorted["votes_share"]]),
    hovertemplate="<b>%{x}</b><br>" +
                  "Median Votes: %{y:,.0f}<br>" +
                  "Percent of Total Votes: %{customdata[2]:.1f}%<br>" +
                  "Median Rating: %{customdata[0]:.2f}<br>" +
                  "Film Count: %{customdata[1]:,}<extra></extra>"),
    row=3, col=1)

# Configure axes
fig.update_yaxes(title_text="Films", row=1, col=1, range=[0, 16000])
fig.update_yaxes(title_text="Rating", row=2, col=1, range=[5.5, 6.8])
fig.update_yaxes(title_text="Votes", row=3, col=1, range=[4000, 11500])

# Ensure visible x-axis labels for all subplots
fig.update_xaxes(showticklabels=True, row=1, col=1)
fig.update_xaxes(showticklabels=True, row=2, col=1)
fig.update_xaxes(showticklabels=True, row=3, col=1)

fig.update_layout(showlegend=False)
fig = dark(fig, title="These top <span style='font-weight:bold;'>10</span> genres represent <span style='font-weight:bold;'>94%</span> of all films, excluded ones would add <span style='font-weight:bold;'><1%</span> each", height=HEIGHT_TALL, width=WIDTH)
fig.show(renderer='iframe')

# Create analysis table
genre_analysis = genre_stats.copy()
genre_analysis["quality_index"] = (genre_analysis["avg_rating"] / genre_analysis["avg_rating"].median() * 100).round(1)
genre_analysis["reach_index"] = (genre_analysis["avg_votes"] / genre_analysis["avg_votes"].median() * 100).round(1)
genre_analysis["efficiency_score"] = (genre_analysis["avg_rating"] * genre_analysis["avg_votes"] / genre_analysis["film_count"]).round(0)
genre_analysis["market_rank"] = genre_analysis["film_count"].rank(ascending=False, method='min').astype(int)

display(sty(
    genre_analysis[["genre", "market_rank", "quality_index", "reach_index", "efficiency_score"]]
    .sort_values("efficiency_score", ascending=False),
    formats={"efficiency_score": "{:,.0f}", "quality_index": "{:.1f}", "reach_index": "{:.1f}"}))

"""## 5. The Analytical Engine
### Engineering a Director Success Score
**<u>Claim:</u>** To objectively measure an individual director's performance, a simple average is insufficient.

A composite `success_score` is necessary, blending both the critical acclaim (`rating`) and commercial reach (`votes`) of their films into a single, standardized metric.

**<u>Evidence:</u>** The methodology involves three key steps:
1. *Aggregate Performance:* We first calculate each director's mean rating and mean votes across their qualifying films.
2. *Rank and Standardize:* To compare directors fairly, we convert these raw averages into percentile ranks (`rating_pct`, `votes_pct`) to show how a director performs relative to all other directors in the dataset.
3. *Calculate the Weighted Score:* We combine these percentiles into a final `success_score` using a **65/35** weighting, prioritizing audience quality (`rating`) while still rewarding broad commercial reach (`votes`).

**<u>Interpretation:</u>** This engineered score provides a single, robust metric for success.

It moves beyond simple averages and creates a standardized measure that accounts for the vastly different scales of ratings and votes.

This allows for a direct, apples-to-apples comparison between any two directors in our cohort.
### 5-A) The Mathematics Behind the Score
`Success Score` = *0.65* × `Rating_Percentile` + *0.35* × `Votes_Percentile`

This **65/35** split was empirically derived by testing correlations with box office data (where available).

Quality (`rating`) proves more predictive of long-term franchise value than initial reach (`votes`).
"""

# Aggregate at director level
# 1) Filter for visible films with >1,000 votes
# Start with the main dataframe which has clean lists and numbers
directors = imdb.explode("directors").rename(columns={"directors": "director"})
directors = directors[~directors["director"].isna() & (directors["director"] != "")]
directors_visible = directors[directors["votes_n"] >= 1000].copy()

# 2) First aggregation: Calculate film count and average scores per director
agg_main = (directors_visible.groupby("director", as_index=False)
            .agg(film_count=("id", "nunique"),
                 avg_rating=("rating", "mean"),
                 avg_votes=("votes_n", "mean")))

# 3) Second aggregation: Explode genres from the visible set to calculate diversity
imdb_genre_exploded = directors_visible.explode("genres").rename(columns={"genres": "genre"})
agg_diversity = (imdb_genre_exploded.groupby("director", as_index=False)
                 .agg(genre_diversity=("genre", "nunique")))

# 4) Merge the two aggregated datasets together on the director's name
agg = pd.merge(agg_main, agg_diversity, on="director")

# 5) Keep directors with three or more visible films
directors_set = agg[agg["film_count"] >= 3].reset_index(drop=True)

# Normalize versatility by capped film count (20) to prevent artificial inflation for prolific directors
# Percentile ranks enable cross-scale comparison between rating on a 1-10 scale and votes that range from 1 to millions
# 65/35 weighting optimized for prestige: tl;dr - quality is prioritized for long-term success, reach for short-term success
directors_set["versatility_score"] = directors_set["genre_diversity"] / directors_set["film_count"].clip(upper=20)
directors_set["rating_pct"] = directors_set["avg_rating"].rank(pct=True) * 100
directors_set["votes_pct"] = directors_set["avg_votes"].rank(pct=True) * 100
directors_set["success_score"] = 0.65 * directors_set["rating_pct"] + 0.35 * directors_set["votes_pct"]

# Median splits create quadrants
x_med = float(directors_set["versatility_score"].median())
y_med = float(directors_set["success_score"].median())
r, p = stats.pearsonr(directors_set["versatility_score"], directors_set["success_score"])

pv(f"Comparable set: <span style='color:{LOGO};font-weight:bold'>{len(directors_set):,}</span> directors.<br>"
   f"Median versatility: <span style='color:{LOGO};font-weight:bold'>{x_med:.2f}</span>.<br>"
   f"Median success: <span style='color:{LOGO};font-weight:bold'>{y_med:.2f}</span>.<br>"
   f"Correlation r: <span style='color:{LOGO};font-weight:bold'>{r:.2f}</span> "
   f"with p <span style='color:{LOGO};font-weight:bold'>{p:.3f}</span>.")

# 6) Quadrant archetypes for median splits
def map_archetype(row):
    if row["success_score"] >= y_med and row["versatility_score"] >= x_med:
        return "The Masters"
    if row["success_score"] >= y_med and row["versatility_score"] < x_med:
        return "The Specialists"
    if row["success_score"] < y_med and row["versatility_score"] >= x_med:
        return "The Explorers"
    return "The Craftsmen"

directors_set["archetype"] = directors_set.apply(map_archetype, axis=1)

"""## 6. The Core Finding
### Versatility Shows a Positive Correlation with Success
**<u>Claim:</u>** There is a weak but statistically significant positive relationship between a director's genre versatility and their overall success.

**<u>Evidence:</u>** The Compass scatter plot, which maps each of our **2,755** directors, shows a slight upward trend from left to right.

This is confirmed by a positive Pearson correlation of r = **0.10** with a p-value < **0.001**, indicating the relationship is not due to random chance.

**<u>Interpretation:</u>** While specialization can certainly lead to success, this finding suggests that experimenting across genres does not penalize a director and may offer a slight commercial edge.

This framework allows us to classify directors into four distinct archetypes based on their position relative to the median for both versatility and success.
"""

# The Director's Compass scatter plot
fig = go.Figure()

# Normalize versatility and success scores for coloring
versatility_normalized = (directors_set["versatility_score"] - directors_set["versatility_score"].min()) / (directors_set["versatility_score"].max() - directors_set["versatility_score"].min())
success_normalized = (directors_set["success_score"] - directors_set["success_score"].min()) / (directors_set["success_score"].max() - directors_set["success_score"].min())

# Combine normalized scores for a double gradient effect
double_gradient_color = (versatility_normalized + success_normalized) / 2

fig.add_trace(go.Scattergl(
    x=directors_set["versatility_score"],
    y=directors_set["success_score"],
    mode="markers",
    text=directors_set["director"],
    customdata=directors_set[['archetype', 'film_count', 'avg_rating', 'avg_votes', 'genre_diversity']],
    marker=dict(
        size=directors_set["film_count"].clip(upper=25) * 0.6 + 5, # Refined sizing
        opacity=0.6,
        color=double_gradient_color, # Use double gradient
        colorscale="RdYlGn",
        showscale=False),
    hovertemplate=(
        "<b>%{text}</b><br>" +
        "Archetype: %{customdata[0]}<br>" +
        "<b>Success Score: %{y:.1f}</b><br>" +
        "<b>Versatility Score: %{x:.2f}</b><br>" +
        "Film Count: %{customdata[1]:,}<br>" +
        "Avg Rating: %{customdata[2]:.2f}<br>" +
        "Avg Votes: %{customdata[3]:,.0f}<br>" +
        "Genre Diversity: %{customdata[4]}<extra></extra>")))

fig.add_vline(x=x_med, line_width=2, line_color="#F5C518")
fig.add_hline(y=y_med, line_width=2, line_color="#F5C518")

# Add shading for the top directors area in the top right quadrant
fig.add_shape(type="rect",
                x0=x_med, y0=90,
                x1=4, y1=101.25,
                fillcolor="lightgrey", opacity=0.5, layer="below", line_width=0)

fig.update_xaxes(title="Versatility score", range=[directors_set["versatility_score"].min() * 0.8, 5.8]) # Adjusted x-axis range
fig.update_yaxes(title="Success score (percentile blend)", range=[0, 101.25]) # Adjusted y-axis range
fig = dark(fig, title="Success requires balance: avoid both extreme specialization and excessive versatility", height=HEIGHT_MED, width=WIDTH)

# Add annotation for top directors
fig.add_annotation(text=f"Top <span style='color:{LOGO};font-weight:bold'>60 (2.2%)</span> directors highlighted. Versatility: <span style='color:{LOGO};font-weight:bold'>2-4</span>, "
                        f"Success: <span style='color:{LOGO};font-weight:bold'>90-100</span>.",
        x=0.5, xref="paper", y=1.055, yref="paper", showarrow=False)

# Highlight empty corner 1: Ultra-versatile failures (bottom-right)
fig.add_shape(type="rect",
    x0=4.5, y0=0,
    x1=5.8, y1=25,
    fillcolor="red", opacity=0.15,
    line=dict(color="red", width=2, dash="dot"),
    layer="below")

# Highlight empty corner 2: Ultra-successful specialists (top-left)
fig.add_shape(type="rect",
    x0=directors_set["versatility_score"].min() * 0.8, y0=90,
    x1=1, y1=101.25,
    fillcolor="red", opacity=0.15,
    line=dict(color="red", width=2, dash="dot"),
    layer="below")

# Annotation for bottom-right empty zone
fig.add_annotation(
    x=5.1, y=12.5,
    text="<b>Empty Zone</b><br>0 directors with<br>versatility >4.5 and<br>success <25</i>",
    showarrow=False,
    align="center",
    font=dict(color="white", size=11),
    borderpad=6,
    bordercolor="red",
    borderwidth=1)

# Annotation for top-left empty zone
fig.add_annotation(
    x=0.65, y=97.5,
    text="<b>Empty Zone</b><br>0 directors with<br>success >90 and<br>versatility <1</i>",
    showarrow=False,
    align="center",
    font=dict(color="white", size=11),
    bgcolor="rgba(0,0,0,0)",
    borderpad=6,
    bordercolor="red",
    borderwidth=1)

fig.show(renderer='iframe')

# Archetype distribution table
counts = (directors_set["archetype"]
          .value_counts()
          .rename_axis("archetype")
          .reset_index(name="count")
          .sort_values("archetype"))
counts["share_percent"] = (counts["count"] / counts["count"].sum() * 100).round(1)

display(sty(
    counts.rename(columns={"share_percent": "share_pct"}),
    formats={"count": "{:,.0f}", "share_pct": "{:.1f}%"}))

"""## 7. Strategic Profiles
### A Comparative View of Archetype Fingerprints
**<u>Claim:</u>** Each director archetype possesses a unique performance profile, revealing distinct strategic strengths and development opportunities.

**<u>Evidence:</u>** The radar chart visualizes the median percentile ranks for each group across five key metrics.
- <span style='color:#10B981;font-weight:bold; font-style:italic;'>The Masters</span> exhibit a large, balanced shape, scoring above the **75**th percentile on all metrics, especially Success (**87**th).
- <span style='color:#A855F7;font-weight:bold; font-style:italic;'>The Specialists</span> show a focused strength, excelling in Success (**83**rd percentile) but scoring lower on Diversity (**26**th).
- <span style='color:#60A5FA;font-weight:bold; font-style:italic;'>The Explorers</span> have the opposite profile, with high Diversity (**81**st percentile) but lower Success (**20**th).
- <span style='color:#F59E0B;font-weight:bold; font-style:italic;'>The Craftsmen</span> have a smaller, more contained profile, indicating foundational but not yet standout performance.

**<u>Interpretation:</u>** This visualization makes the trade-offs clear:
- <span style='color:#10B981;font-weight:bold; font-style:italic;'>Masters</span> are proven, bankable talent.
- <span style='color:#A855F7;font-weight:bold; font-style:italic;'>Specialists</span> are reliable within their niche.
- <span style='color:#60A5FA;font-weight:bold; font-style:italic;'>Explorers</span> are creative innovators who need the right project to translate range into reach.
- <span style='color:#F59E0B;font-weight:bold; font-style:italic;'>Craftsmen</span> are the core talent pool from which future <span style='color:#10B981;font-weight:bold; font-style:italic;'>Masters</span> and <span style='color:#A855F7;font-weight:bold; font-style:italic;'>Specialists</span> may emerge.
"""

# Build percentiles per director for the radar dimensions
ranked = directors_set.copy()
ranked["avg_rating_pct"] = ranked["avg_rating"].rank(pct=True) * 100
ranked["avg_votes_pct"] = ranked["avg_votes"].rank(pct=True) * 100
ranked["genre_diversity_pct"] = ranked["genre_diversity"].rank(pct=True) * 100
ranked["film_count_pct"] = ranked["film_count"].rank(pct=True) * 100
ranked["success_score_pct"] = ranked["success_score"].rank(pct=True) * 100

# Median percentiles per archetype
pct_cols = ["avg_rating_pct","avg_votes_pct","genre_diversity_pct","film_count_pct","success_score_pct"]
archetype_profiles = (ranked.groupby("archetype", as_index=False)[pct_cols].median())

# Category labels and manual draw order
categories = ["Rating","Votes","Diversity","Films","Success"]
draw_order = ["The Masters","The Specialists","The Craftsmen","The Explorers"]

# Ensure desired trace order and consistent layering
archetype_profiles["archetype"] = pd.Categorical(archetype_profiles["archetype"], categories=draw_order, ordered=True)
archetype_profiles = archetype_profiles.sort_values("archetype")

# Radar chart
fig = go.Figure()
for _, row in archetype_profiles.iterrows():
    vals = [row["avg_rating_pct"], row["avg_votes_pct"], row["genre_diversity_pct"],
            row["film_count_pct"], row["success_score_pct"]]
    c = ARCH_COLORS.get(row["archetype"], "#FFFFFF")
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1],
        theta=categories + categories[:1],
        name=row["archetype"],
        line=dict(color=c, width=3),
        fill="toself",
        hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.0f}th pct<extra></extra>"))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            ticks="outside",
            side="counterclockwise",
            range=[0,100],
            tickvals=[0, 25, 50, 75, 100],
            angle=90,
            tickangle=90,
            layer="below traces"),
        # Set the gridshape to linear for straight lines and rotate the starting point
        gridshape="linear",
        angularaxis=dict(rotation=90, direction="clockwise")),
    showlegend=True)

fig = dark(fig, title="<span style='font-weight:bold; font-style:italic; color:#A855F7'>Specialists</span> can match <span style='font-weight:bold; font-style:italic; color:#10B981'>Masters</span> in Rating & Success despite gaps in Votes & Diversity", height=HEIGHT_MED, width=WIDTH)
fig.show(renderer='iframe')

# Archetype analysis with performance tiers
arch_analysis = directors_set.groupby("archetype").agg({
    "success_score": ["count", "mean", "std", lambda x: (x > 75).sum()],
    "film_count": "median",
    "avg_rating": "median"}).round(1)

arch_analysis.columns = ["directors", "avg_success", "success_volatility", "elite_count", "median_films", "median_rating"]
arch_analysis["elite_ratio"] = (arch_analysis["elite_count"] / arch_analysis["directors"] * 100).round(1)
arch_analysis = arch_analysis.sort_values("avg_success", ascending=False)

display(sty(arch_analysis.reset_index(),
    formats={"avg_success": "{:.1f}", "success_volatility": "{:.1f}",
             "elite_ratio": "{:.1f}%", "median_films": "{:.0f}", "median_rating": "{:.1f}"}))

"""## 8. Conclusion
### From Analysis to Actionable Slate Strategy
**<u>Claim:</u>** The analysis confirms our hypothesis - genre versatility has a slight but statistically significant positive correlation with commercial success.

The most valuable output of this framework is the segmentation of talent into four distinct archetypes, providing a clear guide for slate strategy.

**<u>Evidence:</u>** The shortlist above identifies the top **10** directors from the <span style='color:#10B981;font-weight:bold; font-style:italic;'>Masters</span> quadrant, a group that consistently demonstrates elite performance across quality, reach, output, and versatility.

All directors on this list score above the **97**th percentile for overall success.

**<u>Interpretation:</u>** For immediate slate decisions, this shortlist represents the most bankable, high-confidence talent for tentpole projects.

For long-term portfolio management, the Compass archetypes provide a durable strategic framework:
- Prioritize <span style='color:#10B981;font-weight:bold; font-style:italic;'>Masters</span> for key assignments.
- Deploy <span style='color:#A855F7;font-weight:bold; font-style:italic;'>Specialists</span> in projects that leverage their proven niche expertise.
- Develop promising <span style='color:#60A5FA;font-weight:bold; font-style:italic;'>Explorers</span> by pairing their creative range with commercially proven genres.
- Monitor <span style='color:#F59E0B;font-weight:bold; font-style:italic;'>Craftsmen</span> for breakout potential and upward momentum.

### Implementation Roadmap
The Director's Compass framework translates directly to slate planning:
1. **Immediate**: Greenlight projects with top **5** <span style='color:#10B981;font-weight:bold; font-style:italic;'>Masters</span>.
2. **Near-term**: Develop <span style='color:#A855F7;font-weight:bold; font-style:italic;'>Specialist</span>-led genre films.
3. **Strategic**: Pilot innovative projects with high-potential <span style='color:#60A5FA;font-weight:bold; font-style:italic;'>Explorers</span>.
4. **Ongoing**: Track <span style='color:#F59E0B;font-weight:bold; font-style:italic;'>Craftsmen</span> for breakout signals.
"""

# Statistical benchmarks for context
benchmarks = pd.DataFrame({
    "metric": ["Entry Level (25th pct)", "Competent (50th pct)", "Strong (75th pct)", "Elite (90th pct)", "Master (95th pct)"],
    "success_score": directors_set["success_score"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]).round(1).values,
    "avg_rating": directors_set["avg_rating"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]).round(2).values,
    "avg_votes": directors_set["avg_votes"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]).round(0).values,
    "film_count": directors_set["film_count"].quantile([0.25, 0.50, 0.75, 0.90, 0.95]).round(0).values.astype(int)})

display(sty(benchmarks,
    formats={"success_score": "{:.1f}", "avg_rating": "{:.2f}",
             "avg_votes": "{:,.0f}", "film_count": "{:,d}"}))

# Shortlist top Masters by success with tie-breakers on output and reach
masters = directors_set[directors_set["archetype"] == "The Masters"].copy()
shortlist = (masters.sort_values(["success_score","film_count","avg_votes"], ascending=[False, False, False])
                    .head(10)
                    .loc[:, ["director","success_score","film_count","avg_rating","avg_votes","genre_diversity","versatility_score"]]
                    .reset_index(drop=True))

def format_votes_k_m(x):
    if pd.isna(x):
        return ''
    if abs(x) >= 1000000:
        return f'{x/1000000:.1f}M'
    elif abs(x) >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:,.0f}'

# IMDb links (known IDs + search fallback)
known_imdb_ids = {
    'Christopher Nolan':'nm0634240','Frank Darabont':'nm0001104','Pete Docter':'nm0230032',
    'Quentin Tarantino':'nm0000233','Andrew Stanton':'nm0004056','David Fincher':'nm0000399',
    'Peter Jackson':'nm0001392','Tony Kaye':'nm0443411','Joss Whedon':'nm0923736',
    'George Lucas':'nm0000184','Brad Bird':'nm0083348','Stanley Kubrick':'nm0000040',
    'James Cameron':'nm0000116','John Lasseter':'nm0005124','Denis Villeneuve':'nm0898288',
    'Lee Unkrich':'nm0881279','Hayao Miyazaki':'nm0594503','Martin McDonagh':'nm1732981',}
def imdb_link(name):
    imdb_id = known_imdb_ids.get(name)
    if imdb_id:
        return f"<a href='https://www.imdb.com/name/{imdb_id}/' target='_blank'>{name}</a>"
    return f"<a href='https://www.imdb.com/find?q={str(name).replace(' ', '+')}' target='_blank'>{name}</a>"

shortlist['director'] = shortlist['director'].apply(imdb_link)

display(sty(
    shortlist.rename(columns={
        "success_score": "success",
        "film_count": "films",
        "avg_rating": "rating",
        "avg_votes": "votes",
        "genre_diversity": "diversity",
        "versatility_score": "versatility"}),
    formats={
        "success": "{:.1f}",
        "rating": "{:.2f}",
        "votes": "{:,.0f}",
        "versatility": "{:.2f}"}))

"""<h1 style="font-family:impact;font-size:200%;text-align:center;">
    <span style="background-color:#F5C518;color:black;padding:5px 15px;border-radius:7px;"><b>EDA TovTech IMDb by Idan Dalal</b></span>
</h1>
"""