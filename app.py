import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage  # ←ここが重要


load_dotenv("OPENAI_API_KEY")


def generate_answer(user_text: str, expert_type: str) -> str:
    expert_system_messages = {
        "育児アドバイザー": (
            "あなたは育児の専門家です。親の気持ちに寄り添い、具体的で実践的なアドバイスを簡潔に提示してください。"
        ),
        "キャリアコーチ": (
            "あなたはキャリアコーチです。状況を整理し、選択肢と次の具体的アクションを簡潔に提案してください。"
        ),
        "学習サポーター": (
            "あなたは学習方法の専門家です。学習を継続できるよう、短い手順と習慣化のコツを簡潔に提案してください。"
        ),
    }

    system_message = expert_system_messages.get(expert_type, "You are a helpful assistant.")

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_text),
    ]

    result = llm.invoke(messages)
    return result.content



st.set_page_config(page_title="専門家AI相談アプリ", page_icon="🤖")
st.title("🤖 専門家AI相談アプリ（LangChain × Streamlit）")

st.markdown(
    """
このアプリは、入力したテキストを LangChain を通して LLM に渡し、回答を表示します。  
ラジオボタンで「専門家の種類」を選ぶと、LLMの振る舞い（システムメッセージ）が切り替わります。

**使い方**
1. 専門家を選択  
2. 質問を入力  
3. 送信ボタンを押す
"""
)

expert_type = st.radio(
    "専門家の種類を選んでください",
    ["育児アドバイザー", "キャリアコーチ", "学習サポーター"],
    horizontal=True
)

with st.form("question_form"):
    user_text = st.text_area("入力テキスト", placeholder="例：最近ストレスが多いです。")
    submitted = st.form_submit_button("送信")

if submitted:
    if not user_text.strip():
        st.warning("入力テキストを入力してください。")
    else:
        with st.spinner("回答生成中..."):
            try:
                answer = generate_answer(user_text, expert_type)
                st.subheader("回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

st.caption("※ OPENAI_API_KEY は llm.env から読み込みます。")
