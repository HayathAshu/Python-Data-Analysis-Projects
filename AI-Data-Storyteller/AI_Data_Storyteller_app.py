"""
Imarticus Data Science Internship - AI-Powered Data Storyteller
Streamlit app (single-file) that:
- Accepts a CSV upload
- Validates dataset
- Performs automated EDA (summary statistics, correlations, value counts)
- Generates plain-English insights (LLM via HuggingFace if available, else rule-based)
- Creates 3 visualizations (bar, line, heatmap)
- Produces a downloadable PDF report with top insights and plots

How to run:
1. Create a virtualenv (optional but recommended)
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
2. Install requirements:
   pip install -r requirements.txt
   # OR individually:
   pip install streamlit pandas numpy matplotlib seaborn plotly fpdf python-docx transformers
3. Run the app:
   streamlit run AI_Data_Storyteller_app.py

Notes:
- The app will try to use a HuggingFace text2text model for nicer natural-language insights if 'transformers' is installed and the model can be downloaded. If not available, a rule-based summarizer is used.
- The PDF generation uses FPDF and embeds generated plot images.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from io import BytesIO
from fpdf import FPDF

# Try optional imports
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="AI Data Storyteller", layout="wide")

# ---------------------- Helper functions ----------------------

def load_data(uploaded_file):
    name = uploaded_file.name
    try:
        if name.lower().endswith(('.csv')):
            df = pd.read_csv(uploaded_file)
        elif name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            # attempt CSV
            df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def validate_data(df):
    issues = []
    if df is None:
        issues.append('No dataframe provided')
        return {'ok': False, 'issues': issues}
    if df.shape[0] == 0:
        issues.append('Dataset has no rows')
    if df.shape[1] == 0:
        issues.append('Dataset has no columns')
    # missingness
    missing_pct = df.isna().mean().max() * 100
    if missing_pct > 50:
        issues.append(f'More than 50% missing values in at least one column (max {missing_pct:.1f}%)')
    # duplicate rows
    dup = df.duplicated().sum()
    if dup > 0:
        issues.append(f'{dup} duplicate rows found')
    return {'ok': len(issues) == 0, 'issues': issues}


def numeric_summary(df):
    return df.select_dtypes(include=[np.number]).describe().T


def categorical_summary(df, top_n=7):
    cats = df.select_dtypes(include=['object', 'category'])
    summaries = {}
    for col in cats.columns:
        summaries[col] = cats[col].value_counts(dropna=False).head(top_n)
    return summaries


def missing_values_table(df):
    miss = df.isnull().sum()
    miss_pct = (miss / len(df)) * 100
    table = pd.DataFrame({'missing': miss, 'missing_pct': miss_pct})
    table = table.sort_values('missing', ascending=False)
    return table


def top_correlations(df, n=5, threshold=0.3):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    # Select upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr'})
    )
    pairs = pairs.sort_values('corr', ascending=False)
    # filter by threshold
    pairs = pairs[pairs['corr'] >= threshold]
    return pairs.head(n)


def choose_categorical_for_bar(df):
    cats = df.select_dtypes(include=['object', 'category'])
    if cats.shape[1] == 0:
        return None
    # choose column with most non-null values and reasonable unique count
    candidates = [(col, cats[col].nunique(), cats[col].count()) for col in cats.columns]
    # prefer columns with nunique < 100
    candidates = sorted(candidates, key=lambda x: (x[1] > 100, -x[2], x[1]))
    return candidates[0][0]


def choose_timeseries_and_numeric(df):
    # find datetime-like column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            datetime_col = col
            break
    else:
        # attempt to parse any column
        datetime_col = None
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notnull().sum() > 0.5 * len(df):
                    df[col] = parsed
                    datetime_col = col
                    break
            except Exception:
                continue
    # choose numeric column with largest variance
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    y_col = None
    if len(nums) > 0:
        y_col = df[nums].std().sort_values(ascending=False).index[0]
    return datetime_col, y_col


def save_bar_plot(df, col, path):
    plt.figure(figsize=(8,4))
    vc = df[col].value_counts(dropna=False).head(20)
    sns.barplot(x=vc.values, y=vc.index)
    plt.title(f"Top values in {col}")
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_line_plot(df, x_col, y_col, path):
    plt.figure(figsize=(10,4))
    tmp = df[[x_col, y_col]].dropna()
    tmp = tmp.sort_values(x_col)
    plt.plot(tmp[x_col], tmp[y_col], marker='o')
    plt.title(f"{y_col} over {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_correlation_heatmap(df, path):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        # create an empty figure to avoid crashing
        fig = plt.figure(figsize=(4,3))
        plt.text(0.5, 0.5, 'Not enough numeric columns for heatmap', ha='center')
        plt.axis('off')
        fig.savefig(path)
        plt.close()
        return
    corr = num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def llm_summarize(bullets_text):
    """Try to use a small HuggingFace model to craft nicer English insights. Falls back to input text."""
    if TRANSFORMERS_AVAILABLE:
        try:
            # Use a text2text pipeline - flan-t5-small is a common lightweight choice
            generator = pipeline('text2text-generation', model='google/flan-t5-small')
            prompt = (
                "You are a helpful data analyst. Convert the following EDA bullet points into a concise executive summary (5-8 sentences) and 3 short action items.\n\n" + bullets_text
            )
            out = generator(prompt, max_length=256, truncation=True)
            text = out[0]['generated_text']
            return text
        except Exception as e:
            # If model download or runtime fails, fallback
            return None
    return None


def build_rule_based_summary(df):
    bullets = []
    bullets.append(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    miss = missing_values_table(df)
    top_miss = miss[miss['missing']>0].head(3)
    if not top_miss.empty:
        for ix, row in top_miss.iterrows():
            bullets.append(f"Column '{ix}' has {int(row['missing'])} missing values ({row['missing_pct']:.1f}%).")
    else:
        bullets.append('No missing values detected in the top 3 columns.')
    # top correlations
    pairs = top_correlations(df, n=5, threshold=0.5)
    if len(pairs) > 0:
        for _, r in pairs.iterrows():
            bullets.append(f"High correlation ({r['corr']:.2f}) between {r['feature_1']} and {r['feature_2']}")
    else:
        bullets.append('No strong correlations (abs>0.5) found between numeric features.')
    # categorical
    cats = df.select_dtypes(include=['object', 'category'])
    if cats.shape[1] > 0:
        for col in cats.columns[:3]:
            top = cats[col].value_counts(dropna=False).head(1)
            if len(top) > 0:
                bullets.append(f"Top value for '{col}' is '{top.index[0]}' representing {top.values[0]} rows.")
    # skewed numeric
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] > 0:
        skewed = nums.skew().abs().sort_values(ascending=False).head(3)
        for col, val in skewed.items():
            bullets.append(f"Numeric column '{col}' shows skewness {val:.2f}.")
    # put into summary
    summary = ' '.join(bullets[:8])
    actions = "1) Investigate missing values and impute or drop as appropriate. 2) Consider transforming skewed numeric features. 3) Explore highly correlated features for multicollinearity." 
    return summary + "\n\nAction items:\n" + actions


def generate_insights(df):
    rule_summary = build_rule_based_summary(df)
    llm_result = llm_summarize(rule_summary)
    if llm_result:
        return llm_result
    else:
        return rule_summary


def generate_pdf_report(title, insights_text, image_paths, output_path):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(4)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "Executive summary:")
    pdf.ln(2)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, insights_text)

    # Add plots
    for path in image_paths:
        if not os.path.exists(path):
            continue
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, os.path.basename(path), 0, 1)
        # Fit image to page width
        pdf.image(path, x=10, y=25, w=190)

    pdf.output(output_path)
    return output_path

# ---------------------- Streamlit app layout ----------------------

st.title("🧭 Imarticus — AI-Powered Data Storyteller")
st.markdown("Upload a CSV file and get automated EDA, visualizations, natural-language insights, and a downloadable report.")

with st.sidebar:
    st.header('Upload & Settings')
    uploaded_file = st.file_uploader('Upload CSV/Excel file', type=['csv','xls','xlsx'])
    show_raw = st.checkbox('Show raw data head', value=True)
    use_llm = st.checkbox('Try to use local LLM (HuggingFace) for nicer text', value=False)
    generate_report = st.button('Generate PDF Report')

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader('Preview')
    if show_raw and df is not None:
        st.dataframe(df.head(10))

    st.subheader('Validation')
    v = validate_data(df)
    if v['ok']:
        st.success('Basic validations passed')
    else:
        for issue in v['issues']:
            st.warning(issue)

    st.subheader('Quick Statistics')
    with st.expander('Numeric summary (describe)'):
        num_sum = numeric_summary(df)
        st.dataframe(num_sum)
    with st.expander('Missing values (top)'):
        miss_table = missing_values_table(df)
        st.dataframe(miss_table.head(20))
    with st.expander('Categorical top values'):
        cat_summ = categorical_summary(df)
        for col, vc in cat_summ.items():
            st.write(f"**{col}**")
            st.write(vc)

    st.subheader('Automated Insights')
    with st.spinner('Generating insights...'):
        insights = generate_insights(df)
    st.markdown(insights)

    st.subheader('Visualizations')
    # prepare tmp images
    tmpdir = tempfile.mkdtemp()
    images = []
    # bar
    cat_col = choose_categorical_for_bar(df)
    if cat_col:
        bar_path = os.path.join(tmpdir, 'bar.png')
        save_bar_plot(df, cat_col, bar_path)
        images.append(bar_path)
        st.image(bar_path, caption=f'Bar chart: {cat_col}', use_column_width=True)
    else:
        st.info('No categorical column found for bar chart.')

    # line
    datetime_col, y_col = choose_timeseries_and_numeric(df)
    if datetime_col and y_col:
        line_path = os.path.join(tmpdir, 'line.png')
        save_line_plot(df, datetime_col, y_col, line_path)
        images.append(line_path)
        st.image(line_path, caption=f'Line chart: {y_col} over {datetime_col}', use_column_width=True)
    else:
        # fallback: plot top numeric distribution as a line of sorted values
        nums = df.select_dtypes(include=[np.number])
        if nums.shape[1] > 0:
            fallback_col = nums.columns[0]
            line_path = os.path.join(tmpdir, 'line.png')
            plt.figure(figsize=(8,3))
            plt.plot(np.sort(df[fallback_col].dropna().values))
            plt.title(f'Sorted values of {fallback_col}')
            plt.tight_layout()
            plt.savefig(line_path)
            plt.close()
            images.append(line_path)
            st.image(line_path, caption=f'Line chart (fallback): {fallback_col}', use_column_width=True)
        else:
            st.info('No numeric column found for line chart.')

    # heatmap
    heat_path = os.path.join(tmpdir, 'heatmap.png')
    save_correlation_heatmap(df, heat_path)
    images.append(heat_path)
    st.image(heat_path, caption='Correlation heatmap', use_column_width=True)

    # Download PDF report
    if generate_report:
        out_path = os.path.join(tmpdir, 'report.pdf')
        title = f"AI Data Storyteller Report - {uploaded_file.name}"
        with st.spinner('Generating PDF...'):
            pdf_path = generate_pdf_report(title, insights, images, out_path)
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            st.success('Report generated')
            st.download_button('Download PDF report', data=pdf_bytes, file_name='report.pdf', mime='application/pdf')

    # also provide a quick export of the insights text
    st.download_button('Download insights (txt)', data=insights, file_name='insights.txt')

else:
    st.info('Upload a dataset to get started')

# ---------------------- Footer ----------------------
st.markdown('---')
st.caption('This app is a starter template for the Imarticus Data Science internship assessment.')