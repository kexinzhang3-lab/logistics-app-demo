import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. 语言设置与切换 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'  # 默认中文

def toggle_language():
    if st.session_state.language == 'zh':
        st.session_state.language = 'en'
    else:
        st.session_state.language = 'zh'

st.button("🌐 中/En", on_click=toggle_language)

# 文本字典
text = {
    'zh': {
        'title': '物流网络优化与成本计算',
        'factory_cost': '🏭 工厂建设成本 (元)',
        'transport_cost': '🚚 单位运输成本 (元/公里)',
        'distance': '📏 运输距离 (公里)',
        'calc_btn': '计算总成本',
        'result': '💰 总成本: ',
        'detail': '其中: 运输 {} + 建设 {}',
        'viz': '📊 网络可视化 (数字靠近工厂端)',
        'factory': '工厂',
        'customer': '客户'
    },
    'en': {
        'title': 'Logistics Network Optimization',
        'factory_cost': '🏭 Factory Construction Cost ($)',
        'transport_cost': '🚚 Unit Transport Cost ($/km)',
        'distance': '📏 Distance (km)',
        'calc_btn': 'Calculate Total Cost',
        'result': '💰 Total Cost: ',
        'detail': 'Transport {} + Construction {}',
        'viz': '📊 Network Visualization (Labels near source)',
        'factory': 'Factory',
        'customer': 'Customer'
    }
}
lang = text[st.session_state.language]

st.title(lang['title'])

# --- 2. 输入参数 (增加了工厂成本) ---
col1, col2 = st.columns(2)
with col1:
    factory_build_cost = st.number_input(lang['factory_cost'], value=10000)
    transport_unit_cost = st.number_input(lang['transport_cost'], value=5.0)
with col2:
    distance = st.number_input(lang['distance'], value=100)

# --- 3. 计算逻辑 ---
if st.button(lang['calc_btn']):
    transport_total = transport_unit_cost * distance
    total_cost = transport_total + factory_build_cost
    
    st.success(f"{lang['result']}{total_cost}")
    st.info(lang['detail'].format(transport_total, factory_build_cost))

    # --- 4. 可视化 (数字靠近左/开头) ---
    st.subheader(lang['viz'])
    
    G = nx.DiGraph()
    G.add_edge('Factory', 'Customer', weight=distance)
    
    pos = {'Factory': (0, 0), 'Customer': (2, 0)} # 简单的左右布局
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # 画节点
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_family='sans-serif')
    
    # 画边的标签 (关键修改：label_pos=0.2 让数字靠近发出端/开头)
    edge_labels = {('Factory', 'Customer'): f"{distance} km"}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.2, font_color='red')
    
    st.pyplot(fig)