# ui/style.py
import streamlit as st

def hide_anchor_links():
    """Streamlit の見出しアンカー（鎖アイコン）を全て非表示にする"""
    st.markdown("""
    <style>
    a.anchor-link {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


def disable_heading_anchors():
    """h1〜h6 に付くアンカー（鎖アイコン）をアプリ全体で無効化＆非表示にする。"""
    st.markdown(r"""
    <style>
      /* まずCSSで非表示＆クリック不能にする（将来のDOM変更にも多少強い） */
      :where(h1,h2,h3,h4,h5,h6) a[href^="#"] {
        display: none !important;
        pointer-events: none !important;
        visibility: hidden !important;
      }
      /* 旧クラス名対策 */
      a.anchor-link {
        display: none !important;
        pointer-events: none !important;
        visibility: hidden !important;
      }
    </style>
    <script>
      // さらにJSでDOMから物理的に除去（Streamlit再レンダリングにも対応）
      const removeAnchors = () => {
        try {
          const sels = [
            "h1 a[href^='#']","h2 a[href^='#']","h3 a[href^='#']",
            "h4 a[href^='#']","h5 a[href^='#']","h6 a[href^='#']",
            "a.anchor-link"
          ];
          document.querySelectorAll(sels.join(",")).forEach(a => a.remove());
        } catch (_) {}
      };
      // 初回
      removeAnchors();
      // 再レンダリング監視（タブ切替・ウィジェット操作・ページ遷移など）
      new MutationObserver(removeAnchors).observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
   
