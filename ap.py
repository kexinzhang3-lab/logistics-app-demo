import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

st.set_page_config(page_title="Logistics Master Suite", layout="wide")

# --- 全局语言设置 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

# 侧边栏：模块选择
st.sidebar.title("📦 物流决策支持系统")
app_mode = st.sidebar.radio("选择功能模块 / Select Module", 
    ["1. 选址与运输优化 (Location-Transport)", 
     "2. EOQ 库存管理 (Inventory)", 
     "3. 车辆路径规划 (VRP)"])

st.sidebar.button("🌐 中/En", on_click=toggle_language)
st.sidebar.markdown("---")

# ==================================================
# 模块 1: 选址优化 (您之前的代码)
# ==================================================
def app_location():
    tr = {
        'zh': {'title': "🏭 工厂选址与运输优化", 'calc': "开始计算", 'success': "最优方案已找到！"},
        'en': {'title': "🏭 Facility Location Optimization", 'calc': "Optimize", 'success': "Optimal Solution Found!"}
    }
    t = tr[st.session_state.language]
    
    st.header(t['title'])
    
    # 简化的参数输入 (为了节省篇幅，保留核心逻辑)
    col1, col2 = st.columns(2)
    with col1:
        num_factories = st.slider("工厂数量 (F)", 1, 5, 3)
        build_cost = st.number_input("单厂建设成本", value=5000)
    with col2:
        num_customers = st.slider("客户数量 (D)", 1, 5, 3)
        demand_val = st.number_input("默认单客户需求", value=50)

    factory_names = [f"F{i+1}" for i in range(num_factories)]
    customer_names = [f"D{j+1}" for j in range(num_customers)]
    
    # 简单的成本矩阵生成
    costs = pd.DataFrame(
        [[10 + abs(i-j)*2 for j in range(num_customers)] for i in range(num_factories)],
        index=factory_names, columns=customer_names
    )
    st.write("运输单价矩阵:")
    edited_costs = st.data_editor(costs, use_container_width=True)

    if st.button(t['calc'], key='btn_loc'):
        # 简化版 MIP 模型
        prob = pulp.LpProblem("Location", pulp.LpMinimize)
        flow = pulp.LpVariable.dicts("Flow", (factory_names, customer_names), 0, None, pulp.LpInteger)
        is_open = pulp.LpVariable.dicts("Open", factory_names, cat='Binary')
        
        # 目标
        prob += pulp.lpSum([flow[f][d] * edited_costs.loc[f,d] for f in factory_names for d in customer_names]) + \
                pulp.lpSum([is_open[f] * build_cost for f in factory_names])
        
        # 约束
        for d in customer_names:
            prob += pulp.lpSum([flow[f][d] for f in factory_names]) >= demand_val
        for f in factory_names: # 简单的大M产能约束
            prob += pulp.lpSum([flow[f][d] for d in customer_names]) <= 99999 * is_open[f]
            
        prob.solve()
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            st.success(f"{t['success']} 总成本: {pulp.value(prob.objective)}")
            
            # 绘图
            G = nx.DiGraph()
            pos = {}
            for i, f in enumerate(factory_names):
                if is_open[f].varValue > 0.5:
                    G.add_node(f, layer=0, color='gold')
                    pos[f] = (0, -i)
            for i, d in enumerate(customer_names):
                G.add_node(d, layer=1, color='lightgreen')
                pos[d] = (1, -i)
                
            edge_labels = {}
            for f in factory_names:
                for d in customer_names:
                    val = flow[f][d].varValue
                    if val and val > 0:
                        G.add_edge(f, d)
                        edge_labels[(f,d)] = int(val)
            
            fig, ax = plt.subplots()
            colors = [G.nodes[n]['color'] for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            st.pyplot(fig)
        else:
            st.error("无解")

# ==================================================
# 模块 2: EOQ 库存管理
# ==================================================
def app_eoq():
    st.header("📦 EOQ 库存管理计算器")
    st.info("经典经济订货批量模型 (Economic Order Quantity)")
    
    c1, c2, c3 = st.columns(3)
    D = c1.number_input("年需求量 (D)", value=10000)
    S = c2.number_input("单次订货成本 (S)", value=50)
    H = c3.number_input("单位持有成本 (H)", value=2.5)
    
    if st.button("计算 EOQ", key='btn_eoq'):
        # 核心公式
        eoq = math.sqrt((2 * D * S) / H)
        orders_per_year = D / eoq
        total_cost = (D/eoq)*S + (eoq/2)*H
        
        st.metric("最佳订货量 (Q*)", f"{int(eoq)} 件")
        st.metric("年总库存成本", f"¥ {total_cost:,.2f}")
        
        # 锯齿图可视化
        t = np.linspace(0, 10, 1000)
        # 模拟库存随时间变化：Inventory = Q - (DemandRate * t) % Q
        # 这是一个简单的周期函数模拟
        period = 12 / orders_per_year # 周期（月）
        y = [eoq - (x % period) * (eoq/period) for x in t]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, y, color='purple')
        ax.set_title("Inventory Level over Time")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Inventory Units")
        ax.fill_between(t, y, color='purple', alpha=0.1)
        st.pyplot(fig)

# ==================================================
# 模块 3: 车辆路径规划 (VRP)
# ==================================================
def app_vrp():
    st.header("🚚 车辆路径规划 (CVRP)")
    st.caption("目标：使用最少的车辆，走最短的路，服务所有客户。")
    
    # 输入参数
    num_nodes = st.slider("客户数量", 3, 8, 5) # 手机端保持规模小一点，算得快
    vehicle_cap = st.number_input("车辆最大载重", value=50)
    
    # 随机生成坐标和需求
    np.random.seed(42)
    coords = np.random.rand(num_nodes + 1, 2) * 100 # +1 是仓库
    demands = np.random.randint(5, 20, size=num_nodes + 1)
    demands[0] = 0 # 仓库需求为0
    
    # 显示数据表格
    data_df = pd.DataFrame(coords, columns=['X', 'Y'])
    data_df['Type'] = ['Depot'] + ['Customer'] * num_nodes
    data_df['Demand'] = demands
    st.dataframe(data_df.T)
    
    if st.button("规划路径", key='btn_vrp'):
        # 距离矩阵
        dist_matrix = np.zeros((num_nodes+1, num_nodes+1))
        for i in range(num_nodes+1):
            for j in range(num_nodes+1):
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        # PuLP 模型 (简化版 VRP)
        prob = pulp.LpProblem("VRP", pulp.LpMinimize)
        
        # 变量 x[i][j] = 1 代表车从 i 开到 j
        x = pulp.LpVariable.dicts("x", (range(num_nodes+1), range(num_nodes+1)), cat='Binary')
        # 变量 u[i] 用于消除子回路 (MTZ 约束)
        u = pulp.LpVariable.dicts("u", range(num_nodes+1), 0, vehicle_cap, pulp.LpInteger)
        
        # 目标：最小化总距离
        prob += pulp.lpSum([dist_matrix[i][j] * x[i][j] for i in range(num_nodes+1) for j in range(num_nodes+1)])
        
        # 约束
        for i in range(1, num_nodes+1):
            prob += pulp.lpSum([x[i][j] for j in range(num_nodes+1) if i != j]) == 1 # 每个客户被访问一次
            prob += pulp.lpSum([x[j][i] for j in range(num_nodes+1) if i != j]) == 1 # 每个客户离开一次
            
        # MTZ 约束 (消除子回路 + 容量限制)
        for i in range(1, num_nodes+1):
            for j in range(1, num_nodes+1):
                if i != j:
                    prob += u[i] - u[j] + vehicle_cap * x[i][j] <= vehicle_cap - demands[j]
        
        prob.solve()
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            st.success(f"路径规划完成！总距离: {pulp.value(prob.objective):.2f}")
            
            # 绘图
            fig, ax = plt.subplots(figsize=(6, 6))
            # 画点
            ax.scatter(coords[0,0], coords[0,1], c='red', s=200, marker='*', label='Depot')
            ax.scatter(coords[1:,0], coords[1:,1], c='blue', s=100, label='Customers')
            
            # 画线
            for i in range(num_nodes+1):
                for j in range(num_nodes+1):
                    if i != j and x[i][j].varValue > 0.5:
                        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 'k-', alpha=0.6)
                        # 画箭头方向
                        mid_x = (coords[i][0] + coords[j][0]) / 2
                        mid_y = (coords[i][1] + coords[j][1]) / 2
                        ax.text(mid_x, mid_y, '>', fontsize=15, color='gray')

            for i, txt in enumerate(range(num_nodes+1)):
                ax.annotate(f"{txt}({demands[i]})", (coords[i,0]+1, coords[i,1]+1))
                
            plt.legend()
            st.pyplot(fig)
        else:
            st.error("计算超时或无解 (尝试增加车辆载重)")

# ==================================================
# 主程序入口
# ==================================================
if app_mode.startswith("1"):
    app_location()
elif app_mode.startswith("2"):
    app_eoq()
elif app_mode.startswith("3"):
    app_vrp()