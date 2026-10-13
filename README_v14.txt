AI Equity Dashboard v14

新增：
1) Latest Company News：增加 Importance（高/中/低）与 Event Type，并支持筛选。
2) 个股相对表现：个股 + S&P 500 + Nasdaq 100 + 所属AI板块，统一归一化为100，放在同一图中。
3) EPS Consensus Revision：FY1/FY2 EPS 一致预期，显示 Current / 7 / 30 / 60 / 90 days ago。
4) AI Sector Heatmap：按板块市值决定面积，按 当日 / 1M / 3M / YTD 回报着色。
5) 板块 Breadth：上涨/下跌、等权当日、市值加权当日、Above MA20 / MA50 / MA200。

部署：
- 覆盖 app.py / data_layer.py / requirements.txt
- python -m py_compile app.py data_layer.py config.py
- git add app.py data_layer.py requirements.txt
- git commit -m "Add research signals and news classification"
- git push origin main
