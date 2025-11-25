import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="物流运输优化器", layout="wide")

st.title("🏭 物流运输优化与可视化系统")
st.markdown("---")

# 1. 参数设置 (使用侧边栏保持界面整洁)
with st.sidebar:
    st.header("⚙️ 参数设置")
    num_factories = st.slider("工厂数量 (F)", min_value=1, max_value=5, value=3)
    num_customers = st.slider("需求地数量 (D)", min_value=1, max_value=5, value=3)
    
    factory_names = [f"F{i+1}" for i in range(num_factories)]
    customer_names = [f"D{j+1}" for j in range(num_customers)]

# 2. 数据输入
col1, col2 = st.columns(2)

with col1:
    st.subheader("工厂产能 (Supply)")
    supply_data = {}
    for i, f_name in enumerate(factory_names):
        supply_data[f_name] = st.number_input(f"工厂 {f_name} 最大产能", value=100, key=f"s_{i}", min_value=0)

with col2:
    st.subheader("目的地需求 (Demand)")
    demand_data = {}
    for i, d_name in enumerate(customer_names):
        demand_data[d_name] = st.number_input(f"目的地 {d_name} 需求量", value=80, key=f"d_{i}", min_value=0)

st.subheader("运输单价矩阵 (Cost Matrix)")
# 默认运费矩阵
default_costs = [[10 + (i + j) * 2 for j in range(num_customers)] for i in range(num_factories)]
cost_matrix_df = pd.DataFrame(
    default_costs,
    index=factory_names,
    columns=customer_names
)
# 使用 data_editor 允许用户直接修改表格数据
edited_costs = st.data_editor(cost_matrix_df, num_rows="dynamic")

# 3. 核心计算逻辑
if st.button("开始计算最优方案并可视化"):
    # 确保成本矩阵被正确读取
    try:
        cost_df = edited_costs.astype(float)
    except:
        st.error("请确保运费矩阵中的所有值都是有效的数字！")
        st.stop()
        
    # 【修复 NameError 的关键代码】
    prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize) 
    flow = pulp.LpVariable.dicts("Route", (factory_names, customer_names), 0, None, pulp.LpInteger) 
    # 【修复 NameError 的关键代码结束】

    # 目标函数：最小化总运费
    prob += pulp.lpSum([flow[f][d] * cost_df.loc[f, d] for f in factory_names for d in customer_names])

    # 约束条件 1: 产能限制
    for f in factory_names:
        prob += pulp.lpSum([flow[f][d] for d in customer_names]) <= supply_data[f], f"Supply_{f}_Constraint"
    
    # 约束条件 2: 需求满足
    for d in customer_names:
        prob += pulp.lpSum([flow[f][d] for f in factory_names]) >= demand_data[d], f"Demand_{d}_Constraint"

    # 求解
    prob.solve()

    # 4. 结果展示与绘图
    if pulp.LpStatus[prob.status] == 'Optimal':
        st.success(f"✅ 找到最优方案！最低总费用: {pulp.value(prob.objective):.2f}")
        
        G = nx.DiGraph()
        pos = {}
        edge_labels = {}
        
        # 节点位置：工厂在左(x=0)，目的地在右(x=1)
        for i, f in enumerate(factory_names):
            G.add_node(f, layer=0)
            pos[f] = (0, -i * 2) 
        
        for i, d in enumerate(customer_names):
            G.add_node(d, layer=1)
            pos[d] = (1, -i * 2)
            
        # 添加连线（只画有运输量的线）
        for f in factory_names:
            for d in customer_names:
                amount = flow[f][d].varValue
                if amount > 0:
                    G.add_edge(f, d)
                    edge_labels[(f, d)] = f"{int(amount)}"

        # 绘图配置
        fig, ax = plt.subplots(figsize=(10, max(num_factories, num_customers) * 2))
        nx.draw_networkx_nodes(G, pos, node_color=['#ADD8E6' if G.nodes[n]['layer'] == 0 else '#90EE90' for n in G.nodes()], 
                               node_size=3000, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, arrowsize=30, width=2) 
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=14, ax=ax)
        
        plt.axis('off')
        st.pyplot(fig)
        
    else:
        st.error("❌ 无法找到可行解！请检查是否总产能小于总需求，或约束条件过于严格。")
