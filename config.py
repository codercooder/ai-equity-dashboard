from collections import OrderedDict

SECTORS = OrderedDict({
    "1. AI计算芯片 / ASIC / CPU架构": [
        ("NVDA", "NVIDIA"),
        ("AMD", "Advanced Micro Devices"),
        ("AVGO", "Broadcom"),
        ("MRVL", "Marvell Technology"),
        ("ARM", "Arm Holdings"),
        ("INTC", "Intel"),
    ],
    "2. HBM / 存储": [
        ("MU", "Micron Technology"),
        ("SNDK", "SanDisk"),
        ("WDC", "Western Digital"),
        ("SKHY", "SK hynix ADR"),
    ],
    "3. 晶圆代工 / 半导体制造设备": [
        ("TSM", "Taiwan Semiconductor"),
        ("AMAT", "Applied Materials"),
        ("LRCX", "Lam Research"),
        ("KLAC", "KLA"),
    ],
    "4. 网络 / 光通信": [
        ("ANET", "Arista Networks"),
        ("COHR", "Coherent"),
        ("LITE", "Lumentum"),
    ],
    "5. AI服务器": [
        ("DELL", "Dell Technologies"),
        ("HPE", "Hewlett Packard Enterprise"),
        ("SMCI", "Super Micro Computer"),
    ],
    "6. 数据中心基础设施": [
        ("VRT", "Vertiv"),
        ("ETN", "Eaton"),
    ],
    "7. AI电力 / 能源": [
        ("CEG", "Constellation Energy"),
        ("VST", "Vistra"),
        ("GEV", "GE Vernova"),
        ("NRG", "NRG Energy"),
        ("BE", "Bloom Energy"),
    ],
    "8. 数据中心地产": [
        ("EQIX", "Equinix"),
        ("DLR", "Digital Realty"),
    ],
    "9. 云计算 / Hyperscaler": [
        ("MSFT", "Microsoft"),
        ("AMZN", "Amazon"),
        ("GOOGL", "Alphabet"),
        ("ORCL", "Oracle"),
    ],
    "10. AI软件 / Agent": [
        ("PLTR", "Palantir"),
        ("NOW", "ServiceNow"),
        ("CRM", "Salesforce"),
        ("SNOW", "Snowflake"),
    ],
    "11. AI数据 / 开发基础设施": [
        ("DDOG", "Datadog"),
        ("MDB", "MongoDB"),
        ("ESTC", "Elastic"),
        ("GTLB", "GitLab"),
    ],
    "12. AI网络安全": [
        ("PANW", "Palo Alto Networks"),
        ("CRWD", "CrowdStrike"),
        ("ZS", "Zscaler"),
    ],
    "13. 消费端 / Edge AI": [
        ("AAPL", "Apple"),
        ("META", "Meta Platforms"),
        ("QCOM", "Qualcomm"),
    ],
    "14. Physical AI": [
        ("TSLA", "Tesla"),
        ("MBLY", "Mobileye"),
        ("TER", "Teradyne"),
        ("ISRG", "Intuitive Surgical"),
    ],
})

ALL_TICKERS = [ticker for members in SECTORS.values() for ticker, _ in members]
NAME_MAP = {ticker: name for members in SECTORS.values() for ticker, name in members}
