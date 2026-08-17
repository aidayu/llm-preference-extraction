---
description: 結論を起草して Issue を閉じる
allowed-tools: Read, Bash(gh issue view:*), Bash(gh issue comment:*), Bash(gh issue close:*), Bash(gh issue edit:*), Bash(gh pr create:*)
---

対象 Issue: $ARGUMENTS

1. `gh issue view <番号> --comments` で Issue 全体を読む。
2. Issue 本文の「受け入れ条件」または「結果の読み方」を抽出する。
3. 実際の結果（コメント、`reports/<番号>-*/` の中身）を確認する。
4. **事前に書かれた判定基準に照らして**結論を起草する。
   基準を後から書き換えて結果に合わせることは絶対にしない。
   仮説が支持されなかった場合、それを正直に結論として書く。
5. 結論の草案を提示し、承認を求める。
6. 承認されたら:
   - `gh issue comment` で「## 結論」を投稿
   - 仮説が支持されなかった場合は `gh issue edit <番号> --add-label negative-result`
   - コード変更がある場合は PR を作成（本文に `Closes #<番号>`）し、
     マージはユーザーに委ねる
   - コード変更がない場合は `gh issue close <番号>` する
