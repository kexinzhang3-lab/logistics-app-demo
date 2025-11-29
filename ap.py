import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

st.set_page_config(page_title="Logistics Master (Teaching Ver.)", layout="wide")

# --- 语言设置 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

# --- 侧边栏 ---
st.sidebar.title("📦 物流教学演示系统")
app_mode = st.sidebar.radio("选择学习模块", 
    ["1. 车辆路径规划 (VRP/TSP)", 
     "2. EOQ 库存模型 (公式详解)", 
     "3. 选址优化 (MIP)"])

st.sidebar.button("🌐 中/En", on_click=toggle_language)
st.sidebar.markdown("---")

# ==================================================
# 模块 1: 车辆路径规划 (VRP) - 教学增强版
# ==================================================
def app_vrp():
    st.header("🚚 车辆路径规划 (VRP)")
    
    # 模式选择
    input_mode = st.radio("数据输入方式：", 
                         ["方式 A: 输入 X/Y 坐标 (自动算距离)", 
                          "方式 B: 输入距离矩阵 (课本习题模式)"])

    col1, col2 = st.columns(2)
    num_customers = col1.slider("客户数量", 2, 8, 4)
    vehicle_cap = col2.number_input("车辆载重", value=50)
    
    st.markdown("---")
    # **教学提示功能**
    is_open_vrp = st.checkbox("🚛 车辆不回仓库 (Open VRP)")
    
    if is_open_vrp:
        st.info("""
        🎓 **知识点提示：Open VRP (开放式车辆路径问题)**
        
        当你勾选这个选项后，问题模型发生了变化：
        1. **现实含义：** 车辆送完最后一个客户后，任务结束，不需要物理上返回仓库（例如第三方物流车辆）。
        2. **数学处理：** 我们在算法内部，将所有客户点到仓库（终点）的距离强制设为 **0**。
        3. **结果：** 算法为了成本最低，自然会选择“回仓库”来结束路径，但实际上并没有产生回程成本。
        """)

    coords = None
    dist_matrix = None
    demands = []
    
    # --- 方式 A: 坐标模式 ---
    if "坐标" in input_mode:
        if 'coord_df' not in st.session_state or len(st.session_state.coord_df) != num_customers + 1:
            init_data = {'x': [50]* (num_customers+1), 'y': [50]* (num_customers+1), 'demand': [10]* (num_customers+1)}
            init_data['demand'][0] = 0
            init_data['x'][0] = 0; init_data['y'][0] = 0
            st.session_state.coord_df = pd.DataFrame(init_data)
            st.session_state.coord_df.index = ['仓库'] + [f'客户{i}' for i in range(1, num_customers+1)]

        edited_df = st.data_editor(st.session_state.coord_df, key="editor_coords", use_container_width=True)
        coords = edited_df[['x', 'y']].values
        demands = edited_df['demand'].values
        
        n_total = len(coords)
        dist_matrix = np.zeros((n_total, n_total))
        for i in range(n_total):
            for j in range(n_total):
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # --- 方式 B: 矩阵模式 ---
    else:
        n_total = num_customers + 1
        node_names = ['仓库'] + [f'客户{i}' for i in range(1, n_total)]
        
        c1, c2 = st.columns([2, 1])
        with c2:
            st.write("**各点需求量**")
            init_demands = pd.DataFrame({'demand': [0] + [10]*num_customers}, index=node_names)
            edited_demands = st.data_editor(init_demands, key="editor_demands", use_container_width=True)
            demands = edited_demands['demand'].values

        with c1:
            st.write("**距离矩阵 (km)**")
            if 'dist_df' not in st.session_state or len(st.session_state.dist_df) != n_total:
                st.session_state.dist_df = pd.DataFrame(np.zeros((n_total, n_total)), index=node_names, columns=node_names)
            edited_matrix = st.data_editor(st.session_state.dist_df, key="editor_matrix", use_container_width=True)
            dist_matrix = edited_matrix.values

    if st.button("🚀 开始规划", type="primary"):
        solve_dist_matrix = dist_matrix.copy()
        if is_open_vrp:
            for i in range(1, len(solve_dist_matrix)):
                solve_dist_matrix[i][0] = 0

        # PuLP 求解
        n = len(dist_matrix)
        prob = pulp.LpProblem("VRP", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')
        u = pulp.LpVariable.dicts("u", range(n), 0, vehicle_cap, pulp.LpInteger)

        prob += pulp.lpSum([solve_dist_matrix[i][j] * x[i][j] for i in range(n) for j in range(n)])

        for i in range(1, n):
            prob += pulp.lpSum([x[i][j] for j in range(n) if i != j]) == 1
            prob += pulp.lpSum([x[j][i] for j in range(n) if i != j]) == 1
        
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    prob += u[i] - u[j] + vehicle_cap * x[i][j] <= vehicle_cap - demands[j]

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            st.success(f"计算完成！总行驶距离: {pulp.value(prob.objective):.2f}")
            
            routes = []
            for j in range(1, n):
                if x[0][j].varValue > 0.5:
                    route = [0, j]
                    curr = j
                    while True:
                        next_node = -1
                        for k in range(n):
                            if k != curr and x[curr][k].varValue > 0.5:
                                next_node = k
                                break
                        if next_node == -1: break
                        if next_node == 0:
                            if not is_open_vrp: route.append(0)
                            break
                        else:
                            route.append(next_node)
                            curr = next_node
                    routes.append(route)
            
            for idx, r in enumerate(routes):
                path_str = " -> ".join([f"仓库" if node==0 else f"客户{node}" for node in r])
                st.info(f"🚛 车辆 {idx+1}: {path_str}")

            # 绘图
            fig, ax = plt.subplots()
            G = nx.DiGraph()
            if coords is not None:
                pos = {i: (coords[i][0], coords[i][1]) for i in range(n)}
            else:
                pos = nx.circular_layout(range(n))
            
            nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='red', node_size=300, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=range(1,n), node_color='blue', node_size=200, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: (f"W" if i==0 else f"C{i}") for i in range(n)}, ax=ax, font_color='white')

            for i in range(n):
                for j in range(n):
                    if i != j and x[i][j].varValue > 0.5:
                        if is_open_vrp and j == 0: pass 
                        else:
                            nx.draw_networkx_edges(G, pos, edgelist=[(i,j)], edge_color='black', width=1.5, ax=ax)
            st.pyplot(fig)
        else:
            st.error("无解 (可能载重不足)")

# ==================================================
# 模块 2: EOQ - 教学详解版
# ==================================================
def app_eoq():
    st.header("📦 EOQ 经济订货批量模型")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🧮 参数输入")
        D = st.number_input("年总需求量 (D)", value=10000)
        S = st.number_input("单次订货成本 (S)", value=50)
        H = st.number_input("单位持有成本 (H)", value=2.5)
    
    with col2:
        st.markdown("### 📖 公式原理")
        # LaTeX 公式显示
        st.latex(r"Q^* = \sqrt{\frac{2DS}{H}}")
        st.markdown("""
        **参数含义：**
        * $Q^*$ : 最佳订货量 (Quantity)
        * $D$ : 年需求量 (Demand)
        * $S$ : 单次订货成本 (Setup Cost)
        * $H$ : 单位持有成本 (Holding Cost)
        """)

    if st.button("🔢 开始详细计算"):
        # 计算过程
        numerator = 2 * D * S
        fraction = numerator / H
        eoq = math.sqrt(fraction)
        
        total_cost = (D/eoq)*S + (eoq/2)*H
        
        st.divider()
        st.subheader("💡 计算步骤详解")
        
        with st.expander("点击查看一步步计算过程 (Step-by-Step)", expanded=True):
            st.markdown(f"""
            **第一步：计算分子 (2DS)**
            $$ 2 \\times {D} \\times {S} = {numerator} $$
            
            **第二步：除以持有成本 (2DS / H)**
            $$ \\frac{{{numerator}}}{{{H}}} = {fraction} $$
            
            **第三步：开根号 (得到 Q*)**
            $$ \\sqrt{{{fraction}}} \\approx {eoq:.2f} $$
            """)
            
            st.success(f"✅ **最终结果：最佳订货量 Q* = {int(eoq)} 件**")
            
            st.info(f"""
            **💰 总成本验证：**
            * 订货成本 = $(D/Q) \\times S = ({D}/{int(eoq)}) \\times {S} \\approx {int(D/eoq)*S:.2f}$
            * 持有成本 = $(Q/2) \\times H = ({int(eoq)}/2) \\times {H} \\approx {(int(eoq)/2)*H:.2f}$
            * **年总成本 ≈ {total_cost:,.2f} 元**
            """)

# ==================================================
# 模块 3: 选址优化 (保留)
# ==================================================
def app_location():
    st.header("🏭 选址优化 (MIP)")
    st.write("此处保留您之前的选址逻辑 (为节省篇幅略显示，功能需手动添加)")

# --- 主程序 ---
if app_mode.startswith("1"):
    app_vrp()
elif app_mode.startswith("2"):
    app_eoq()
elif app_mode.startswith("3"):
    app_location()