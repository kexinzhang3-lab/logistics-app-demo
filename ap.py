import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Logistics Location Optimizer", layout="wide")

# --- 1. 语言设置 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

tr = {
    'zh': {
        'title': "🏭 物流选址与运输路径联合优化",
        'sidebar': "网络规模设置",
        'n_factories': "备选工厂数量 (F)",
        'n_customers': "需求地数量 (D)",
        'factory_settings': "🏭 备选工厂参数 (产能 & 建设成本)",
        'cap_label': "最大产能",
        'fixed_cost_label': "建设成本",
        'dem_title': "🏢 目的地需求",
        'dem_label': "需求量",
        'cost_matrix': "🚚 运输单价矩阵 (单位: 元/个)",
        'btn_calc': "🚀 计算最优选址与运输方案",
        'err_num': "请输入有效的数字！",
        'success': "🎉 找到最优方案！",
        'decision_title': "🏗️ 选址决策结果",
        'open': "✅ 建设",
        'close': "❌ 不建",
        'metrics_title': "💰 成本构成分析",
        'total_cost': "总综合成本",
        'trans_cost': "运输总费用",
        'build_cost': "建设总费用",
        'no_solution': "无解！即使所有工厂都建也无法满足总需求。",
        'viz_title': "📊 网络可视化 (仅显示建设的工厂)"
    },
    'en': {
        'title': "🏭 Facility Location & Transport Optimization",
        'sidebar': "Network Size",
        'n_factories': "Potential Factories (F)",
        'n_customers': "Customers (D)",
        'factory_settings': "🏭 Factory Parameters (Cap & Fixed Cost)",
        'cap_label': "Capacity",
        'fixed_cost_label': "Fixed Cost",
        'dem_title': "🏢 Customer Demand",
        'dem_label': "Demand",
        'cost_matrix': "🚚 Unit Transport Cost Matrix",
        'btn_calc': "🚀 Optimize Location & Transport",
        'err_num': "Invalid number input!",
        'success': "Optimal Solution Found!",
        'decision_title': "🏗️ Location Decisions",
        'open': "✅ Open",
        'close': "❌ Closed",
        'metrics_title': "💰 Cost Analysis",
        'total_cost': "Total Cost",
        'trans_cost': "Transport Cost",
        'build_cost': "Construction Cost",
        'no_solution': "Infeasible! Total capacity < Total demand.",
        'viz_title': "📊 Network Visualization (Opened Factories Only)"
    }
}
t = tr[st.session_state.language]

# --- 2. 界面布局 ---
col_head, col_btn = st.columns([5, 1])
with col_head:
    st.title(t['title'])
with col_btn:
    st.button("🌐 中/En", on_click=toggle_language)

st.markdown("---")

# 侧边栏：规模
with st.sidebar:
    st.header(t['sidebar'])
    num_factories = st.slider(t['n_factories'], 1, 5, 3)
    num_customers = st.slider(t['n_customers'], 1, 5, 3)
    
    factory_names = [f"F{i+1}" for i in range(num_factories)]
    customer_names = [f"D{j+1}" for j in range(num_customers)]

# 主界面输入
col1, col2 = st.columns(2)

# 工厂参数输入 (现在包含建设成本)
with col1:
    st.subheader(t['factory_settings'])
    supply_data = {}
    fixed_cost_data = {}
    
    for f in factory_names:
        c1, c2 = st.columns(2)
        with c1:
            supply_data[f] = st.number_input(f"{f} {t['cap_label']}", value=100, step=10, key=f"s_{f}")
        with c2:
            # 这里的 Key 必须唯一，加上 fc_ 前缀
            fixed_cost_data[f] = st.number_input(f"{f} {t['fixed_cost_label']}", value=5000, step=1000, key=f"fc_{f}")

# 需求输入
with col2:
    st.subheader(t['dem_title'])
    demand_data = {}
    for d in customer_names:
        demand_data[d] = st.number_input(f"{d} {t['dem_label']}", value=60, step=10, key=f"d_{d}")

# 运费矩阵
st.subheader(t['cost_matrix'])
default_costs = [[10 + (i + j) * 2 for j in range(num_customers)] for i in range(num_factories)]
cost_df = pd.DataFrame(default_costs, index=factory_names, columns=customer_names)
edited_costs = st.data_editor(cost_df, key="cost_editor", use_container_width=True)

# --- 3. 核心算法：混合整数规划 (MIP) ---
if st.button(t['btn_calc'], type="primary"):
    # 建立模型
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)

    # 变量1：运输量 (连续变量, >=0)
    flow = pulp.LpVariable.dicts("Flow", (factory_names, customer_names), 0, None, pulp.LpInteger)
    
    # 变量2：是否建厂 (0/1 整数变量)
    # 1 代表建设，0 代表不建
    is_open = pulp.LpVariable.dicts("IsOpen", factory_names, cat='Binary')

    # 目标函数：最小化 (运输总成本 + 启用的工厂建设成本)
    transport_cost = pulp.lpSum([flow[f][d] * edited_costs.loc[f, d] for f in factory_names for d in customer_names])
    build_cost = pulp.lpSum([is_open[f] * fixed_cost_data[f] for f in factory_names])
    
    prob += transport_cost + build_cost

    # 约束1：需求必须满足
    for d in customer_names:
        prob += pulp.lpSum([flow[f][d] for f in factory_names]) >= demand_data[d]

    # 约束2：工厂产出不能超过产能，且只有建了厂(is_open=1)才能产出
    for f in factory_names:
        # 如果 is_open[f] 是 0，则右边是 0，意味着该工厂流出量必须是 0
        prob += pulp.lpSum([flow[f][d] for d in customer_names]) <= supply_data[f] * is_open[f]

    # 求解
    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        st.success(t['success'])
        
        # 提取结果
        total_obj = pulp.value(prob.objective)
        total_trans = pulp.value(transport_cost)
        total_build = pulp.value(build_cost)
        
        # --- 显示选址决策 ---
        st.subheader(t['decision_title'])
        cols = st.columns(num_factories)
        opened_factories = []
        
        for i, f in enumerate(factory_names):
            status = is_open[f].varValue
            if status > 0.5: # 选中
                cols[i].success(f"{f}: {t['open']}")
                cols[i].caption(f"💰{fixed_cost_data[f]}")
                opened_factories.append(f)
            else: # 未选中
                cols[i].error(f"{f}: {t['close']}")
                cols[i].caption(f"<s>💰{fixed_cost_data[f]}</s>")

        # --- 成本分析 ---
        st.subheader(t['metrics_title'])
        m1, m2, m3 = st.columns(3)
        m1.metric(t['total_cost'], f"{total_obj:,.2f}")
        m2.metric(t['trans_cost'], f"{total_trans:,.2f}")
        m3.metric(t['build_cost'], f"{total_build:,.2f}")

        # --- 可视化 (只画选中的工厂) ---
        st.subheader(t['viz_title'])
        G = nx.DiGraph()
        pos = {}
        edge_labels = {}
        
        # 布局
        for i, f in enumerate(factory_names):
            # 只有建了的厂才画实色，没建的画虚化或者不画连接
            if f in opened_factories:
                G.add_node(f, layer=0, status='open')
            else:
                G.add_node(f, layer=0, status='closed')
            pos[f] = (0, -i * 1.5)
        
        for i, d in enumerate(customer_names):
            G.add_node(d, layer=1)
            pos[d] = (2, -i * 1.5) # 拉开距离方便看字

        # 边
        for f in factory_names:
            for d in customer_names:
                amount = flow[f][d].varValue
                if amount and amount > 0:
                    G.add_edge(f, d)
                    edge_labels[(f, d)] = f"{int(amount)}"

        fig, ax = plt.subplots(figsize=(8, max(num_factories, num_customers) * 1.5 + 1))
        
        # 画节点颜色
        color_map = []
        for n in G.nodes():
            if G.nodes[n].get('layer') == 1:
                color_map.append('#90EE90') # 客户绿
            elif G.nodes[n].get('status') == 'open':
                color_map.append('#FFD700') # 建厂金
            else:
                color_map.append('#D3D3D3') # 没建厂灰

        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2500, ax=ax, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # 只画有流量的边
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='blue', arrows=True, arrowsize=20, width=1.5, alpha=0.6)
        
        # 标签靠近左侧 (label_pos=0.2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=11, label_pos=0.2, ax=ax, rotate=False)
        
        plt.axis('off')
        st.pyplot(fig)

    else:
        st.error(t['no_solution'])