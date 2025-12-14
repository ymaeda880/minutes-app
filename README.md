streamlit run app.py

pip install -r requirements.txt

brew install ffmpeg

```
your-app/
├─ app.py # ホーム（設定・使い方）
├─ requirements.txt
├─ config.py # API エンドポイント・価格・為替初期値・API キー取得
├─ lib/
│ ├─ **init**.py
│ ├─ audio.py # 音声長推定（mutagen → wave → audioread）
│ └─ costs.py # 料金計算（Chat の USD 概算）
├─ ui/
│ ├─ **init**.py
│ └─ sidebar.py # サイドバー描画 & state 初期化
└─ pages/
├─ 01*文字起こし.py # Whisper ページ（USD/JPY 表示）
└─ 02*議事録作成.py # 議事録ページ（USD/JPY 表示）
```

# ====== 代替プロンプト ======

ALT_PROMPT = """あなたは日本語の会議文字起こしを整形する専門家です。
以下の「生の文字起こしテキスト」を読み、話者を推定しながら発話ごとに改行して可読化してください。

必須要件:

1. 話者ラベルは S1:, S2:, S3: ... の形式（人数が確定できない場合は最小限で開始し、必要に応じて追加）
2. 司会者は発言内容から特定し，話者ラベルを「司会者：」としてください．
3. 文字は一字一句変えないでください．
4. 固有名詞・数値・日付は可能な範囲で正規化
5. タイムスタンプらしき表記（例: [00:12:34]）があれば各発話の先頭に残す。なければ付与不要
6. 出力は以下の 2 部構成:
   【整形テキスト】...（S1:, S2: で始まる行で構成）
   【メモ】話者数の推定, 主要トピック 3 点, 用語ゆらぎの正規化例

注意:

- 文字は変更しない．文字を付け加えない．
- 会話の順序を維持
- 話者ごとの改行して，さらに１行空けてください．
- 「生の文字起こしテキスト」において，文字起こし時の変換のミスと思われる箇所は最後の【メモ】の欄に記載
  """

<br><br>

# 使い方

1. 音声ファイル分割
   - １つが 20 分程度の音声ファイルに分割します．
2. 文字起こし
   - 分割したそれぞれの音声ファイルの文字起こしを行います．
3. （手作業）テキスト結合
   - 文字起こしされたテキストを結合して 1 つのテキストファイルにします．その際にオーバーラップされた部分を取り除きます．
4. 話者分離
   - 結合した 1 つのテキストファイルに対して話者の分離を行います．
   - その結果作成されるテキストファイルを<span style="color: lightblue;">話者分離テキスト</span>と呼ぶことにします．
5. （手作業）逐語録作成
   - 音声を実際に聞きながら，<span style="color: lightblue;">話者分離テキスト</span>の間違いを直します．
   - その結果作成されるファイルが<span style="color: lightblue;">逐語録</span>になります．
6. 議事録作成
   - <span style="color: lightblue;">逐語録</span>から<span style="color: lightblue;">議事録</span>を作成します．

##########

[Errno 2] No such file or directory: 'ffprobe'
「ffprobe が見つからない（実行できない）」というエラー
Streamlit コードには ffprobe を呼ぶ行は一切ありません。
しかし、pydub が内部で ffmpeg / ffprobe を使う場合があります。

🎯 結論：Mac に ffprobe（FFmpeg のセット）が入っていない
FFmpeg を入れれば自動的に ffprobe も入る。

✅ 解決方法（Mac）
1）Homebrew で FFmpeg をインストールする

ターミナルで：
brew install ffmpeg

これだけで OK です。
ffmpeg と ffprobe が両方入ります：

which ffmpeg
which ffprobe
