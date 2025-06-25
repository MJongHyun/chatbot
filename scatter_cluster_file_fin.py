import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'C:\\Windows\\Fonts\\gulim.ttc'
font = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font)

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_path, "fonts", "malgun.ttf") # 'malgun.ttf'는 실제 폰트 파일명으로 변경

    # 2. Matplotlib에 폰트 경로 지정 및 적용
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
    plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지

    st.success(f"폰트를 성공적으로 불러왔습니다: {font_path}")

except FileNotFoundError:
    st.error(f"폰트 파일을 찾을 수 없습니다. '{font_path}' 경로를 확인해주세요.")
    st.info("프로젝트 폴더에 'fonts' 폴더를 만들고 그 안에 .ttf 폰트 파일을 넣었는지, 파일 이름이 정확한지 확인하세요.")
    st.stop() # 폰트가 없으면 앱 실행 중지

# --- 1. 앱 설정 ---

st.set_page_config(layout="wide")
st.title("문서 기반 답변 품질 분석 대시보드")

# --- 2. 데이터 처리 및 시각화 함수 (변경 없음) ---
# 이전과 동일한 함수들을 그대로 사용합니다.

def intent_v1_v2_true_false_res(cbot_res_df):
    df = cbot_res_df.copy()
    idxTFDF = pd.DataFrame()
    for idx in df['index'].tolist():
        mask = df['index'] == idx
        correct_TF_v2, correct_TF_v1 = df.loc[mask, 'correct_TF'].iloc[0], df.loc[mask, 'correct_TF_v1'].iloc[0]
        try:
            correct_TF_v2_list = set(ast.literal_eval(df.loc[mask, 'v2_int_code'].iloc[0]))
        except (ValueError, SyntaxError):
            correct_TF_v2_list = set()
        correct_TF_v1_list = df.loc[mask, 'qu_intent'].iloc[0]
        v1_DF = pd.DataFrame([correct_TF_v1_list], columns=['code']); v1_DF['index'] = idx; v1_DF['div'] = 'V1'; v1_DF['TF'] = correct_TF_v1
        if correct_TF_v2_list:
            v2_DF = pd.DataFrame(list(correct_TF_v2_list), columns=['code']); v2_DF['index'] = idx; v2_DF['div'] = 'V2'; v2_DF['TF'] = correct_TF_v2
            idxTFDF = pd.concat([idxTFDF, v1_DF, v2_DF], axis=0)
        else:
            idxTFDF = pd.concat([idxTFDF, v1_DF], axis=0)
    idxTFDF = idxTFDF.reset_index(drop=True)
    x_res_deno = idxTFDF[['code']].value_counts().reset_index(name='deno')
    x_res_nume = idxTFDF[idxTFDF['TF'] == True][['code']].value_counts().reset_index(name='nume')
    x_nume_deno = pd.merge(x_res_deno, x_res_nume, on='code', how='left').fillna(0)
    return idxTFDF, x_nume_deno

def scatter_plot_res_make(x_nume_deno):
    df = x_nume_deno.copy()
    if not df.empty:
        plot_df_final = df.rename(columns={'code': 'TypeName', 'deno': 'TotalInquiries', 'nume': 'CorrectAnswers'})
        plot_df_final['AccuracyRate'] = np.round(plot_df_final['CorrectAnswers'] / plot_df_final['TotalInquiries'] * 100, 2)
        return plot_df_final.drop_duplicates(subset=['TypeName']).reset_index(drop=True)
    return pd.DataFrame()

def res_excel_df_v1_v2_scatter(_cbot_row_idxTFDF, _cbot_scatter_df, _cbot_res_df, _cbot_answer_df, llm_judge_threshold):
    v2_row_res = _cbot_row_idxTFDF[_cbot_row_idxTFDF['div'] == 'V2']
    v2_all_code = pd.merge(_cbot_res_df, v2_row_res[['code', 'index']].drop_duplicates(), on='index')
    v2_all_code_g = np.round(v2_all_code.groupby('code')['v2_judge_mean_score'].mean(), 2).reset_index(name='all_v2_score')
    v2_code_g = v2_all_code_g.rename(columns={'code': 'TypeName'})
    r1 = pd.merge(_cbot_answer_df, v2_code_g, on='TypeName', how='inner')
    r2 = pd.merge(_cbot_scatter_df, r1, on='TypeName', how='inner')
    
    if r2.empty: return pd.DataFrame()
    median_inquiries = r2['TotalInquiries'].median()
    
    def assign_quadrant(row):
        if row['TotalInquiries'] > median_inquiries and row['all_v2_score'] > llm_judge_threshold: return '1사분면: 많음/높음'
        if row['TotalInquiries'] <= median_inquiries and row['all_v2_score'] > llm_judge_threshold: return '2사분면: 적음/높음'
        if row['TotalInquiries'] <= median_inquiries and row['all_v2_score'] <= llm_judge_threshold: return '3사분면: 적음/낮음'
        if row['TotalInquiries'] > median_inquiries and row['all_v2_score'] <= llm_judge_threshold: return '4사분면: 많음/낮음'
        return 'N/A'
    
    r2['Quadrant'] = r2.apply(assign_quadrant, axis=1)
    r2.rename(columns={'TypeName': '문서코드', 'TotalInquiries': '전체 참고 문서 수', 'CorrectAnswers': 'LLM-Cor 참고 문서 수', 'AccuracyRate': '정확도', 'all_v2_score': 'LLM 평균 LLM Judge', 'document_data': '등록한 문서', 'Quadrant': '사분면유형'}, inplace=True)
    return r2[['사분면유형', '문서코드', '전체 참고 문서 수', 'LLM-Cor 참고 문서 수', '정확도', 'LLM 평균 LLM Judge', '등록한 문서']]

def interactive_scatter_plot(plot_df, service_name, llm_judge_score_median):
    median_inquiries = plot_df['전체 참고 문서 수'].median()
    fig = go.Figure()
    quadrant_order = ['1사분면: 많음/높음', '2사분면: 적음/높음', '3사분면: 적음/낮음', '4사분면: 많음/낮음']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    for quadrant_name, color in zip(quadrant_order, colors):
        subset = plot_df[plot_df['사분면유형'] == quadrant_name]
        if not subset.empty:
            hovertemplate = "<b>%{customdata[0]}</b><br>참고 수: %{x}<br>LLM 점수: %{y:.2f}<br>정확도: %{customdata[1]:.2f}%<extra></extra>"
            fig.add_trace(go.Scatter(x=subset['전체 참고 문서 수'], y=subset['LLM 평균 LLM Judge'], mode='markers', name=quadrant_name, marker=dict(size=12, color=color), customdata=np.stack((subset['문서코드'], subset['정확도']), axis=-1), hovertemplate=hovertemplate))
    fig.add_vline(x=median_inquiries, line_width=1.5, line_dash="dash", line_color="dimgray", annotation_text=f"문서 수 중앙값: {median_inquiries:.0f}")
    fig.add_hline(y=llm_judge_score_median, line_width=1.5, line_dash="dash", line_color="dimgray", annotation_text=f"점수 기준선: {llm_judge_score_median:.2f}")
    fig.update_layout(title=f'<b>{service_name} 문서별 답변 품질 분석</b>', xaxis_title='참고된 문서 수 (개)', yaxis_title='LLM Judge 평균 점수', legend_title_text='<b>사분면 유형</b>', height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- 3. Streamlit 앱 메인 UI 구성 ---

st.sidebar.title("📄 파일 업로드")
# 파일 업로드 로직 변경: 총 4개의 파일 업로드
# cbot_answer_file = st.sidebar.file_uploader("1. 문서-답변 원본 파일 (cbot_answer)", type=['parquet'])
# c_df_file = st.sidebar.file_uploader("2. 성과측정 결과 파일 (c_df)", type=['parquet'])
# qdf_file = st.sidebar.file_uploader("3. 질문 클러스터 분석 파일", type=['xlsx', 'csv', 'parquet'])
# keyword_file = st.sidebar.file_uploader("4. 키워드 분석 파일", type=['xlsx', 'csv', 'parquet'])

cbot_answer_file = './c_bot_code_fin.parquet'
c_df_file =  './c_df_all.parquet'
qdf_file = r'C:\Users\MJH\Downloads\chatbot\chatbot_v2_test\llm_check_plus_rag\c_bot_test_cluster_res_0625.xlsx'
keyword_file = r'C:\Users\MJH\Downloads\chatbot\chatbot_v2_test\llm_check_plus_rag\c_bot_keyword_check_0625.xlsx'

# 헬퍼 함수: 파일 확장자에 따라 데이터프레임 로드
def load_dataframe(file_object):
    if file_object is None:
        return None
    file_name = file_object.name
    if file_name.endswith('.parquet'):
        return pd.read_parquet(file_object)
    elif file_name.endswith('.csv'):
        return pd.read_csv(file_object)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(file_object)
    else:
        st.error(f"지원하지 않는 파일 형식입니다: {file_name}")
        return None

# 4개의 파일이 모두 업로드되었는지 확인
# if cbot_answer_file and c_df_file and qdf_file and keyword_file:
# 1. 파일 로드
try:
    cbot_answer_df = pd.read_parquet(cbot_answer_file).rename(columns={'document_name': 'TypeName'})[['TypeName', 'document_data']]
    c_df = pd.read_parquet(c_df_file)
    qDF_refine = pd.read_excel(qdf_file)
    keywordAllDF = pd.read_excel(keyword_file)
except Exception as e:
    st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 산점도 생성
st.header("📊 문서별 답변 품질 개요")
st.info("아래 산점도는 각 문서의 '참고된 횟수'와 '답변 품질(LLM Judge 점수)'을 시각화한 것입니다.")
col1, col2 = st.columns([1, 3])
with col1:
    llm_judge_score_input = st.number_input(
        label="LLM Judge 점수 기준선 설정", min_value=0.0, max_value=5.0, value=4.20, step=0.05, format="%.2f",
        help="이 값을 변경하면 사분면과 점의 색상이 실시간으로 변경됩니다."
    )
cbot_row_idxTFDF, cbot_idxTFDF_pre = intent_v1_v2_true_false_res(c_df)
cbot_scatter_df_pre = scatter_plot_res_make(cbot_idxTFDF_pre)
final_df = res_excel_df_v1_v2_scatter(cbot_row_idxTFDF, cbot_scatter_df_pre, c_df, cbot_answer_df, llm_judge_score_input)
interactive_scatter_plot(final_df, '챗봇 성능', llm_judge_score_input)

# 3. 상세 분석 결과 표시
st.header("🔍 답변 실패 상세 분석")
st.info("산점도에 표시된 문서 중 하나를 선택하여 미리 분석된 '답변 실패' 상세 결과를 확인합니다.")

if qDF_refine.empty:
    st.warning("상세 분석 결과 데이터가 비어있습니다.")
else:
    doc_options = sorted(final_df['문서코드'].unique().tolist())
    selected_doc = st.selectbox("상세 분석을 원하는 문서 코드를 선택하세요:", options=doc_options, key="doc_selector")
    
    if selected_doc:
        try:
            doc_text = cbot_answer_df[cbot_answer_df['TypeName'] == selected_doc]['document_data'].iloc[0]
            with st.expander("선택한 문서 원본 내용 보기"):
                st.markdown(doc_text)
        except IndexError:
            st.warning("선택한 문서의 원본 내용을 찾을 수 없습니다.")

        st.subheader(f"'{selected_doc}' 문서의 주요 실패 키워드")
        keywords_df = keywordAllDF[keywordAllDF['TypeName'] == selected_doc].sort_values('count', ascending=False).head(10)
        st.dataframe(keywords_df[['keyword', 'count']].rename(columns={'keyword':'키워드', 'count':'언급 횟수'}))

        st.subheader(f"'{selected_doc}' 문서의 실패 질문 클러스터 분석")
        doc_detail_df = qDF_refine[qDF_refine['TypeName'] == selected_doc].copy()
        
        st.markdown("##### 🔢 결과 필터링")
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            unique_clusters = doc_detail_df['cluster'].unique().tolist()
            selected_clusters = st.multiselect("분석 결과 선택", options=unique_clusters, default=unique_clusters)
        with filter_cols[1]:
            min_semantic = st.number_input("의미 유사도 ≥", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
        with filter_cols[2]:
            min_lexical = st.number_input("어휘 유사도 ≥", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
        with filter_cols[3]:
            max_rank = st.number_input("평균 Rank ≤", min_value=0.0, value=10.0, step=0.5, format="%.1f")

        filtered_df = doc_detail_df[
            (doc_detail_df['cluster'].isin(selected_clusters)) &
            (doc_detail_df['semantic_score'] >= min_semantic) &
            (doc_detail_df['lexical_score'] >= min_lexical) &
            (doc_detail_df['question_cluster_rank'] <= max_rank)
        ]

        display_cols = {
            'question_cluster': '유사 질문 클러스터', 'cluster': '분석 결과', 'semantic_score': '의미 유사도', 
            'lexical_score': '어휘 유사도', 'question_cluster_rank': '평균 Rank', 'TRUE_CNT': '정답(T) 수', 'FALSE_CNT': '오답(F) 수'
        }
        display_cols_exist = {k:v for k,v in display_cols.items() if k in filtered_df.columns}
        st.dataframe(filtered_df[display_cols_exist.keys()].rename(columns=display_cols_exist))

# else:
#     st.info("👈 사이드바에서 분석에 필요한 파일 4개를 모두 업로드해주세요.")
