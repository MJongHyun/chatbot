import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from collections import Counter
import matplotlib.pyplot as plt


# --- 1. 앱 설정 ---
st.set_page_config(layout="wide")
st.title("문서 기반 챗봇 답변 분석 대시보드")

# --- 2. 데이터 처리 및 시각화 함수 ---

# 원본 함수 (변경 없음)
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

# 원본 함수 (변경 없음)
def scatter_plot_res_make(x_nume_deno):
    df = x_nume_deno.copy()
    if not df.empty:
        plot_df_final = df.rename(columns={'code': 'TypeName', 'deno': 'TotalInquiries', 'nume': 'CorrectAnswers'})
        plot_df_final['AccuracyRate'] = np.round(plot_df_final['CorrectAnswers'] / plot_df_final['TotalInquiries'] * 100, 2)
        return plot_df_final.drop_duplicates(subset=['TypeName']).reset_index(drop=True)
    return pd.DataFrame()


# [수정됨] 사분면 할당 로직을 분리하고, 두 개의 기준선을 받도록 수정
def res_excel_df_v1_v2_scatter(_cbot_row_idxTFDF, _cbot_scatter_df, _cbot_res_df, _cbot_answer_df, llm_judge_threshold, ref_docs_threshold):
    v2_row_res = _cbot_row_idxTFDF[_cbot_row_idxTFDF['div'] == 'V2']
    v2_all_code = pd.merge(_cbot_res_df, v2_row_res[['code', 'index']].drop_duplicates(), on='index')
    v2_all_code_g = np.round(v2_all_code.groupby('code')['v2_judge_mean_score'].mean(), 2).reset_index(name='all_v2_score')
    v2_code_g = v2_all_code_g.rename(columns={'code': 'TypeName'})
    r1 = pd.merge(_cbot_answer_df, v2_code_g, on='TypeName', how='inner')
    r2 = pd.merge(_cbot_scatter_df, r1, on='TypeName', how='inner')

    if r2.empty: return pd.DataFrame()

    # 사분면 할당 함수 (두 기준선을 사용)
    def assign_quadrant(row):
        if row['TotalInquiries'] > ref_docs_threshold and row['all_v2_score'] > llm_judge_threshold: return '1사분면: 많음/높음'
        if row['TotalInquiries'] <= ref_docs_threshold and row['all_v2_score'] > llm_judge_threshold: return '2사분면: 적음/높음'
        if row['TotalInquiries'] <= ref_docs_threshold and row['all_v2_score'] <= llm_judge_threshold: return '3사분면: 적음/낮음'
        if row['TotalInquiries'] > ref_docs_threshold and row['all_v2_score'] <= llm_judge_threshold: return '4사분면: 많음/낮음'
        return 'N/A'

    r2['Quadrant'] = r2.apply(assign_quadrant, axis=1)
    r2.rename(columns={'TypeName': '문서코드', 'TotalInquiries': '전체 참고 문서 수', 'CorrectAnswers': 'LLM-Cor 참고 문서 수', 'AccuracyRate': '정확도', 'all_v2_score': 'LLM 평균 LLM Judge', 'document_data': '등록한 문서', 'Quadrant': '사분면유형'}, inplace=True)
    return r2[['사분면유형', '문서코드', '전체 참고 문서 수', 'LLM-Cor 참고 문서 수', '정확도', 'LLM 평균 LLM Judge', '등록한 문서']]

# [수정됨] 플롯 함수가 두 개의 기준선 값을 직접 받도록 수정
def interactive_scatter_plot(plot_df, service_name, llm_judge_score_threshold, ref_docs_threshold):
    fig = go.Figure()
    quadrant_order = ['1사분면: 많음/높음', '2사분면: 적음/높음', '3사분면: 적음/낮음', '4사분면: 많음/낮음']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'] # 많/높(초록), 적/높(파랑), 적/낮(주황), 많/낮(빨강)

    for quadrant_name, color in zip(quadrant_order, colors):
        subset = plot_df[plot_df['사분면유형'] == quadrant_name]
        if not subset.empty:
            hovertemplate = "<b>%{customdata[0]}</b><br>참고 문서 수: %{x}<br>LLM 점수: %{y:.2f}<extra></extra>"
            fig.add_trace(go.Scatter(
                x=subset['전체 참고 문서 수'],
                y=subset['LLM 평균 LLM Judge'],
                mode='markers',
                name=quadrant_name,
                marker=dict(size=12, color=color),
                customdata=np.stack((subset['문서코드'], subset['정확도']), axis=-1),
                hovertemplate=hovertemplate
            ))

    # X축과 Y축 기준선을 사용자가 입력한 값으로 설정
    fig.add_vline(x=ref_docs_threshold, line_width=1.5, line_dash="dash", line_color="dimgray", annotation_text=f"문서 수 기준선: {ref_docs_threshold:.0f}")
    fig.add_hline(y=llm_judge_score_threshold, line_width=1.5, line_dash="dash", line_color="dimgray", annotation_text=f"점수 기준선: {llm_judge_score_threshold:.2f}")

    fig.update_layout(
        title=f'<b>{service_name} 문서별 답변 점수 결과 분석</b>',
        xaxis_title='참고된 문서 수 (개)',
        yaxis_title='LLM Judge 평균 점수',
        legend_title_text='<b>사분면 유형</b>',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. 데이터 로드 ---
# 파일 경로 (사용자 환경에 맞게 수정 필요)
try:
    cbot_answer_file = './data/c_bot_code_fin.parquet'
    c_df_file = './data/c_df_all.parquet'
    qdf_file = './data/c_bot_test_cluster_res_0625.xlsx'
    keyword_file = './data/c_bot_keyword_check_0625.xlsx'

    cbot_answer_df = pd.read_parquet(cbot_answer_file).rename(columns={'document_name': 'TypeName'})[['TypeName', 'document_data']]
    c_df = pd.read_parquet(c_df_file)
    qDF_refine = pd.read_excel(qdf_file)
    keywordAllDF = pd.read_excel(keyword_file)
except Exception as e:
    st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# --- 4. 대시보드 UI 및 로직 ---

# 4-1. 산점도 생성
st.header("문서별 답변 평가")
st.info("아래 산점도는 문의에 따른 답변시 각 문서의 '참고된 횟수'와 'LLM Judge 평균 점수'를 시각화합니다. \n 기준선을 조절하여 사분면을 재구성하고, 문서를 그룹별로 확인할 수 있습니다. (초기값 :  X, Y 중앙 값으로 적용)")

# [수정됨] 기준선 설정을 위한 UI와 로직
# (1) 사분면 계산 전 원본 데이터 생성
cbot_row_idxTFDF, cbot_idxTFDF_pre = intent_v1_v2_true_false_res(c_df)
cbot_scatter_df_pre = scatter_plot_res_make(cbot_idxTFDF_pre)

# (2) 초기 데이터프레임 생성을 위해 임시값(0)으로 함수 호출 (사분면 제외 데이터만 필요)
base_df = res_excel_df_v1_v2_scatter(cbot_row_idxTFDF, cbot_scatter_df_pre, c_df, cbot_answer_df, 0, 0)

# (3) 데이터 중앙값을 기준선의 기본값으로 설정
if not base_df.empty:
    ref_docs_median_default = base_df['전체 참고 문서 수'].median()
    llm_judge_score_median_default = base_df['LLM 평균 LLM Judge'].median()
else:
    ref_docs_median_default = 10.0  # 데이터 없을 시 기본값
    llm_judge_score_median_default = 4.2  # 데이터 없을 시 기본값

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("##### 📊 기준선 설정")
    ref_docs_input = st.number_input(
        label="참고된 문서 수 기준선",
        min_value=0,
        value=int(ref_docs_median_default),  # 중앙값으로 기본값 설정
        step=1,
        help="X축 기준선입니다. 이 값을 변경하면 사분면과 점의 색상이 실시간으로 변경됩니다."
    )
    llm_judge_score_input = st.number_input(
        label="LLM Judge 점수 기준선",
        min_value=0.0, max_value=5.0,
        value=float(llm_judge_score_median_default), # 중앙값으로 기본값 설정
        step=0.05,
        format="%.2f",
        help="Y축 기준선입니다. 이 값을 변경하면 사분면과 점의 색상이 실시간으로 변경됩니다."
    )

# (4) 사용자가 설정한 기준선으로 최종 데이터프레임 생성 및 시각화
final_df = res_excel_df_v1_v2_scatter(cbot_row_idxTFDF, cbot_scatter_df_pre, c_df, cbot_answer_df, llm_judge_score_input, ref_docs_input)
interactive_scatter_plot(final_df, '챗봇 성능', llm_judge_score_input, ref_docs_input)


# 4-2. 상세 분석 결과 표시
st.header("불명확한 답변 상세 분석")
st.info("산점도을 통해 분석이 필요한 문서를 찾고, 아래에서 해당 문서 코드를 선택하여 상세 원인을 파악하세요. \n 아래에서 제공하는 결과는 '죄송합니다', '찾을 수 없습니다' 등 불명확한 답변을 제공한 문의와 주요 키워드를 확인할 수 있습니다.")

if final_df.empty:
    st.warning("상세 분석을 위한 데이터가 비어있습니다.")
else:
    doc_options = sorted(final_df['문서코드'].unique().tolist())
    selected_doc = st.selectbox("상세 분석을 원하는 문서 코드를 선택하세요:", options=doc_options, key="doc_selector")

    if selected_doc:
        try:
            doc_text = cbot_answer_df[cbot_answer_df['TypeName'] == selected_doc]['document_data'].iloc[0]
            with st.expander("등록한 문서 원본 내용 보기"):
                st.markdown(f"```\n{doc_text}\n```")
        except IndexError:
            st.warning("등록한 문서의 원본 내용을 찾을 수 없습니다.")

        st.subheader(f"'{selected_doc}' 문서가 참고된 문의에서 추출한 주요 키워드")
        st.info("해당 문서를 참고한 여러 문의 중, 불명확한 답변에 사용된 주요 키워드를 보여드립니다. 문서 수정 시 활용하면 챗봇이 답변을 제공하도록 도움을 줄 수 있습니다.")
        
        # [수정됨] head(10)으로 상위 10개만 필터링하고, dataframe 높이를 지정해 스크롤 생성
        keywords_df = keywordAllDF[keywordAllDF['TypeName'] == selected_doc].sort_values('count', ascending=False).head(10)
        st.dataframe(
            keywords_df[['keyword', 'count']].rename(columns={'keyword': '키워드', 'countㄴ': '언급 횟수'}),
            height=385, # 데이터프레임의 높이를 지정하여 10개 초과 시 스크롤바 생성 (10개에 맞춰 조정)
            use_container_width=False
        )

        st.subheader(f"'{selected_doc}' 문서가 참고된 불명확한 답변 원인 분석")
        st.info("해당 문서를 참고한 여러 문의 중 불명확한 답변에 비슷한 실문의 군집정보, 문의와 참고문서 내용간의 유사도 정보를 제공합니다.")
        st.info("의미유사도도 높거나 어휘유사도도 높은 경우의 문의를 확인하여 문서 수정에 활용하면 챗봇이 답변을 제공하도록 도움을 줄 수 있습니다.")
        doc_detail_df = qDF_refine[qDF_refine['TypeName'] == selected_doc].copy()

        st.markdown("##### 필터링")
        filter_cols = st.columns(4)

        with filter_cols[0]:
            min_semantic = st.number_input("의미 유사도 ≥", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
        with filter_cols[1]:
            min_lexical = st.number_input("어휘 유사도 ≥", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")

        filtered_df = doc_detail_df[
            (doc_detail_df['semantic_score'] >= min_semantic) &
            (doc_detail_df['lexical_score'] >= min_lexical)
        ]

        display_cols = {
            'question_cluster': '유사 실문의 군집 결과', 'semantic_score': '의미 유사도', 'lexical_score': '어휘 유사도'
        }
        display_cols_exist = {k: v for k, v in display_cols.items() if k in filtered_df.columns}
        st.dataframe(filtered_df[display_cols_exist.keys()].rename(columns=display_cols_exist))
