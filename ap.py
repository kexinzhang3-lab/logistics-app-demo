import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

st.set_page_config(page_title="Logistics Master Ultimate", layout="wide")

# --- 1. 语言设置 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

# --- 2. 双语字典 (Translation Dictionary) ---
tr = {
    'zh': {
        'title': "🚛 物流决策支持系统 v4.0",
        'subtitle': "集成运筹优化、库存管理与路径规划的教学平台",
        'sidebar_title': "⚙️ 控制面板",
        'sidebar_info': "请选择功能模块",
        'modules': ["1. 车辆路径规划 (VRP)", "2. EOQ 库存模型", "3. 选址优化 (MIP)"],
        # VRP 模块
        'vrp_title': "🗺️ 车辆路径规划系统",
        'vrp_desc': "通过 :red-background[运筹优化算法] 计算多车辆的最短配送路径。",
        'vrp_mode': "数据输入方式",
        'vrp_modes': ["方式 A: 输入 X/Y 坐标 (地图模式)", "方式 B: 输入距离矩阵 (课本模式)"],
        'vrp_params': "👇在此配置参数",
        'num_cust': "客户数量",
        'veh_cap': "车辆载重",
        'open_vrp': "车辆不回仓库 (Open VRP)",
        'open_vrp_hint': "勾选后，车辆送完最后一个客户直接下班，不再计算回程距离。",
        'btn_plan': "🚀 立即规划路径",
        'res_dist': "总行驶距离",
        'res_veh': "所需车辆",
        'demand_table': "各点需求量",
        'dist_table': "距离矩阵 (km)",
        'coord_table': "坐标列表",
        'no_solution': "无解 (可能载重不足)",
        # EOQ 模块
        'eoq_title': "📦 库存控制中心",
        'tab1': "🧮 计算器",
        'tab2': "📖 公式原理",
        'D': "年需求量 (D)",
        'S': "单次订货成本 (S)",
        'H': "单位持有成本 (H)",
        'btn_calc': "计算 EOQ",
        'eoq_res': "最佳订货量",
        'eoq_desc': "该公式用于平衡订货成本与持有成本。",
        # Location 模块
        'loc_title': "🏭 选址优化",
        'loc_warn': "⚠️ 该模块正在维护中..."
    },
    'en': {
        'title': "🚛 Logistics Decision Support v4.0",
        'subtitle': "Integrated Platform for OR, Inventory & Routing",
        'sidebar_title': "⚙️ Control Panel",
        'sidebar_info': "Select Module",
        'modules': ["1. Vehicle Routing (VRP)", "2. EOQ Model", "3. Facility Location (MIP)"],
        # VRP
        'vrp_title': "🗺️ Vehicle Routing System",
        'vrp_desc': "Optimize routes using :red-background[Operations Research].",
        'vrp_mode': "Input Mode",
        'vrp_modes': ["Mode A: X/Y Coordinates (Map)", "Mode B: Distance Matrix (Textbook)"],
        'vrp_params': "👇 Parameters",
        'num_cust': "Number of Customers",
        'veh_cap': "Vehicle Capacity",
        'open_vrp': "Open VRP (No Return)",
        'open_vrp_hint': "Vehicles end their route at the last customer.",
        'btn_plan': "🚀 Optimize Routes",
        'res_dist': "Total Distance",
        'res_veh': "Vehicles Used",
        'demand_table': "Demands",
        'dist_table': "Distance Matrix (km)",
        'coord_table': "Coordinates",
        'no_solution': "Infeasible (Check Capacity)",
        # EOQ
        'eoq_title': "📦 Inventory Control",
        'tab1': "🧮 Calculator",
        'tab2': "📖 Formula",
        'D': "Annual Demand (D)",
        'S': "Setup Cost (S)",
        'H': "Holding Cost (H)",
        'btn_calc': "Calculate EOQ",
        'eoq_res': "Optimal Order Qty",
        'eoq_desc': "Balances setup costs and holding costs.",
        # Location
        'loc_title': "🏭 Facility Location",
        'loc_warn': "⚠️ Module under maintenance..."
    }
}
t = tr[st.session_state.language]

# --- 3. 顶部 Banner 与 标题 ---
st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title(t['title'])
    st.markdown(f":grey[{t['subtitle']}]")
with col_h2:
    st.button("🌐 中/En", on_click=toggle_language)
st.divider()

# --- 4. 侧边栏 ---
with st.sidebar:
    st.header(t['sidebar_title'])
    st.info(t['sidebar_info'])
    
    # 这里的 options 需要处理一下，只取索引 0,1,2，或者直接用中文列表
    # 简单起见，我们根据语言显示不同列表，但逻辑通过 index 判断
    selected_module_text = st.radio("Nav", t['modules'], label_visibility="collapsed")
    
    # 判断选了第几个（0, 1, 2）
    module_index = t['modules'].index(selected_module_text)

    st.markdown("---")
    st.caption("Powered by Python & Streamlit")

# ==================================================
# 模块 1: VRP (双语 + 美化 + 双模式)
# ==================================================
def app_vrp():
    st.subheader(t['vrp_title'])
    st.markdown(t['vrp_desc'])
    
    input_mode = st.radio(t['vrp_mode'], t['vrp_modes'])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(t['vrp_params'])
        num_customers = st.slider(t['num_cust'], 2, 8, 4)
        vehicle_cap = st.number_input(t['veh_cap'], value=50)
        is_open_vrp = st.checkbox(t['open_vrp'], help=t['open_vrp_hint'])

    coords = None
    dist_matrix = None
    demands = []

    with col2:
        # --- 方式 A: 坐标模式 ---
        if input_mode == t['vrp_modes'][0]: 
            if 'coord_df' not in st.session_state or len(st.session_state.coord_df) != num_customers + 1:
                init_data = {'x': [50]* (num_customers+1), 'y': [50]* (num_customers+1), 'demand': [10]* (num_customers+1)}
                init_data['demand'][0] = 0
                init_data['x'][0] = 0; init_data['y'][0] = 0
                st.session_state.coord_df = pd.DataFrame(init_data)
                st.session_state.coord_df.index = ['W'] + [f'C{i}' for i in range(1, num_customers+1)]

            st.caption(t['coord_table'])
            edited_df = st.data_editor(st.session_state.coord_df, key="editor_coords", use_container_width=True, height=200)
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
            node_names = ['W'] + [f'C{i}' for i in range(1, n_total)]
            
            c_a, c_b = st.columns([1, 2])
            with c_a:
                st.caption(t['demand_table'])
                init_demands = pd.DataFrame({'D': [0] + [10]*num_customers}, index=node_names)
                edited_demands = st.data_editor(init_demands, key="editor_demands", use_container_width=True, height=200)
                demands = edited_demands['D'].values
            with c_b:
                st.caption(t['dist_table'])
                if 'dist_df' not in st.session_state or len(st.session_state.dist_df) != n_total:
                    st.session_state.dist_df = pd.DataFrame(np.zeros((n_total, n_total)), index=node_names, columns=node_names)
                edited_matrix = st.data_editor(st.session_state.dist_df, key="editor_matrix", use_container_width=True, height=200)
                dist_matrix = edited_matrix.values

    if st.button(t['btn_plan'], type="primary"):
        solve_dist_matrix = dist_matrix.copy()
        if is_open_vrp:
            for i in range(1, len(solve_dist_matrix)):
                solve_dist_matrix[i][0] = 0

        # 求解
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
            st.divider()
            # 结果卡片
            col_res1, col_res2 = st.columns(2)
            
            # 统计车辆数
            veh_count = 0
            for j in range(1, n):
                if x[0][j].varValue > 0.5: veh_count += 1
            
            col_res1.metric(t['res_dist'], f"{pulp.value(prob.objective):.2f}")
            col_res2.metric(t['res_veh'], f"{veh_count}")

            # 绘图
            fig, ax = plt.subplots(figsize=(6, 4))
            G = nx.DiGraph()
            if coords is not None:
                pos = {i: (coords[i][0], coords[i][1]) for i in range(n)}
            else:
                pos = nx.circular_layout(range(n))
            
            nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='red', node_size=300, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=range(1,n), node_color='blue', node_size=200, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: (f"W" if i==0 else f"C{i}") for i in range(n)}, ax=ax, font_color='white', font_size=8)

            for i in range(n):
                for j in range(n):
                    if i != j and x[i][j].varValue > 0.5:
                        if is_open_vrp and j == 0: pass 
                        else:
                            nx.draw_networkx_edges(G, pos, edgelist=[(i,j)], edge_color='black', width=1.5, ax=ax, arrowsize=15)
            st.pyplot(fig)
        else:
            st.error(t['no_solution'])

# ==================================================
# 模块 2: EOQ (双语 + 美化)
# ==================================================
def app_eoq():
    st.subheader(t['eoq_title'])
    
    tab1, tab2 = st.tabs([t['tab1'], t['tab2']])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        D_val = c1.number_input(t['D'], 10000)
        S_val = c2.number_input(t['S'], 50)
        H_val = c3.number_input(t['H'], 2.5)
        
        if st.button(t['btn_calc']):
            eoq = int(math.sqrt(2*D_val*S_val/H_val))
            st.balloons()
            st.success(f"{t['eoq_res']}: **{eoq}**")
    
    with tab2:
        st.latex(r"Q^* = \sqrt{\frac{2DS}{H}}")
        st.caption(t['eoq_desc'])

# ==================================================
# 模块 3: 选址 (保留位)
# ==================================================
def app_location():
    st.subheader(t['loc_title'])
    st.warning(t['loc_warn'])

# --- 路由 ---
if module_index == 0:
    app_vrp()
elif module_index == 1:
    app_eoq()
elif module_index == 2:
    app_location()