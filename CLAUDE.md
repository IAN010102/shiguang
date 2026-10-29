# 同步規則:拾光小本 ↔ Grace Studio Hub

這個 App 在總站(`grace-hub`)有一頁介紹:`data/apps/little-finds.json`。

只要這次改動新增、移除或改變了使用者看得到的功能、文字、語言、隱私行為(例如資料送到哪裡)或數量(分頁數、玩法數、範本包數…),完成後請:

1. 更新 `grace-hub/data/apps/little-finds.json`(zh 與 en 都要),必要時重拍 `assets/little-finds/` 的畫面。
2. 在 `grace-hub/data/updates.json` 最前面加一則「最近更新」。
3. 在 `grace-hub` 執行 `python3 tools/build.py`,再執行 `python3 tools/sync_check.py`,直到沒有警告;最後 `python3 tools/sync_check.py --bless`。
4. 兩邊都先 commit,等使用者說「推」才 push。
