import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Logistics Optimizer", layout="wide")

# --- 1. 语言设置 (Language Settings) ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

# 语言字典
tr = {
    'zh': {
        'title': "🏭 物流运输优化与可视化系统",
        'sidebar': "参数设置",
        'n_factories': "工厂数量 (F)",
        'n_customers': "需求地数量 (D)",
        'build_cost': "🏭 单个工厂建设固定成本",
        'cap_title': "工厂产能",
        'cap_label': "最大产能",
        'dem_title': "目的地需求",
        'dem_label': "需求量",
        'cost_matrix': "运输单价矩阵 (点击表格修改)",
        'btn_calc': "开始计算最优方案并可视化",
        'err_num': "请确保运费矩阵中的所有值都是有效的数字！",
        'success': "找到最优方案！",
        'total_cost': "💰 总综合成本",
        'trans_cost': "🚛 运输总费用",
        'const_cost': "🏗️ 建设总费用",
        'no_solution': "无法找到可行解！请检查是否总产能小于总需求。",
        'viz_title': "📊 网络可视化 (数字靠近工厂端)"
    },
    'en': {
        'title': "🏭 Logistics Optimization System",
        'sidebar': "Settings",
        'n_factories': "Number of Factories (F)",
        'n_customers': "Number of Customers (D)",
        'build_cost': "🏭 Construction Cost per Factory",
        'cap_title': "Factory Capacity",
        'cap_label': "Max Capacity",
        'dem_title': "Customer Demand",
        'dem_label': "Demand",
        'cost_matrix': "Unit Transport Cost Matrix (Editable)",
        'btn_calc': "Optimize & Visualize",
        'err_num': "Please ensure all values in the matrix are numbers!",
        'success': "Optimal Solution Found!",
        'total_cost': "💰 Total Integrated Cost",
        'trans_cost': "🚛 Total Transport Cost",
        'const_cost': "🏗️ Total Construction Cost",
        'no_solution': "No solution found! Check if total supply < total demand.",
        'viz_title': "📊 Network Visualization (Labels near Source)"
    }
}
t = tr[st.session_state.language]

# --- 2. 界面布局 (UI Layout) ---
col_head, col_btn = st.columns([5, 1])
with col_head:
    st.title(t['title'])
with col_btn:
    st.button("🌐 中/En", on_click=toggle_language)

st.markdown("---")

with st.sidebar:
    st.header(t['sidebar'])
    num_factories = st.slider(t['n_factories'], min_value=1, max_value=5, value=3)
    num_customers = st.slider(t['n_customers'], min_value=1, max_value=5, value=3)
    # 新增功能：工厂建设成本输入
    st.markdown("---")
    build_cost_per_factory = st.number_input(t['build_cost'], value=5000, step=1000)
    
    factory_names = [f"F{i+1}" for i in range(num_factories)]
    customer_names = [f"D{j+1}" for j in range(num_customers)]

col1, col2 = st.columns(2)

with col1:
    st.subheader(t['cap_title'])
    supply_data = {}
    for i, f_name in enumerate(factory_names):
        supply_data[f_name] = st.number_input(f"{f_name} {t['cap_label']}", value=100, key=f"s_{i}", min_value=0)

with col2:
    st.subheader(t['dem_title'])
    demand_data = {}
    for i, d_name in enumerate(customer_names):
        demand_data[d_name] = st.number_input(f"{d_name} {t['dem_label']}", value=80, key=f"d_{i}", min_value=0)

st.subheader(t['cost_matrix'])
default_costs = [[10 + (i + j) * 2 for j in range(num_customers)] for i in range(num_factories)]
cost_matrix_df = pd.DataFrame(default_costs, index=factory_names, columns=customer_names)
edited_costs = st.data_editor(cost_matrix_df, num_rows="dynamic", use_container_width=True)

# --- 3. 核心计算逻辑 (Core Logic) ---
if st.button(t['btn_calc'], type="primary"):
    try:
        cost_df = edited_costs.astype(float)
    except:
        st.error(t['err_num'])
        st.stop()

    # 建立优化模型
    prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)
    flow = pulp.LpVariable.dicts("Route", (factory_names, customer_names), 0, None, pulp.LpInteger)

    # 目标函数：最小化运输成本
    prob += pulp.lpSum([flow[f][d] * cost_df.loc[f, d] for f in factory_names for d in customer_names])

    # 约束条件
    for f in factory_names:
        prob += pulp.lpSum([flow[f][d] for d in customer_names]) <= supply_data[f]
    
    for d in customer_names:
        prob += pulp.lpSum([flow[f][d] for f in factory_names]) >= demand_data[d]

    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        # 计算各种成本
        transport_optimal_cost = pulp.value(prob.objective)
        total_construction_cost = build_cost_per_factory * num_factories
        grand_total = transport_optimal_cost + total_construction_cost

        # 显示结果
        st.success(f"{t['success']}")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(t['total_cost'], f"{grand_total:,.2f}")
        metric_col2.metric(t['trans_cost'], f"{transport_optimal_cost:,.2f}")
        metric_col3.metric(t['const_cost'], f"{total_construction_cost:,.2f}")
        
        # --- 4. 可视化优化 (Visualization) ---
        st.subheader(t['viz_title'])
        G = nx.DiGraph()
        pos = {}
        edge_labels = {}
        
        # 布局：工厂在左(x=0)，客户在右(x=1)
        for i, f in enumerate(factory_names):
            G.add_node(f, layer=0)
            pos[f] = (0, -i * 1.5)  # 调整间距
        
        for i, d in enumerate(customer_names):
            G.add_node(d, layer=1)
            pos[d] = (1, -i * 1.5)
            
        for f in factory_names:
            for d in customer_names:
                amount = flow[f][d].varValue
                if amount > 0:
                    G.add_edge(f, d)
                    edge_labels[(f, d)] = f"{int(amount)}"

        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(num_factories, num_customers) * 1.5 + 1))
        color_map = ['#ADD8E6' if G.nodes[n]['layer'] == 0 else '#90EE90' for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2500, ax=ax, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, arrowsize=20, width=1.5, alpha=0.7)
        
        # 关键修改：label_pos=0.25 让数字标签靠近左侧(工厂端)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12, label_pos=0.25, ax=ax, rotate=False)
        
        plt.axis('off')
        st.pyplot(fig)
        
    else:
        st.error(t['no_solution'])