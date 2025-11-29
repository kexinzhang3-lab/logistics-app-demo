import streamlit as st
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

st.set_page_config(page_title="Logistics Master v5.1", layout="wide")

# --- 1. 语言设置 ---
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

def toggle_language():
    st.session_state.language = 'en' if st.session_state.language == 'zh' else 'zh'

# --- 2. 双语字典 (已修复 Key) ---
tr = {
    'zh': {
        'title': "🚛 物流决策支持系统 v5.1",
        'subtitle': "集成运筹优化、库存管理与路径规划的综合平台",
        'sidebar_title': "⚙️ 控制面板",
        'modules': ["1. 车辆路径规划 (VRP)", "2. EOQ 库存模型", "3. 选址优化 (MIP)"],
        # VRP
        'vrp_title': "🗺️ 车辆路径规划系统",
        'vrp_mode': "数据输入方式",
        'vrp_modes': ["方式 A: 输入 X/Y 坐标 (地图模式)", "方式 B: 输入距离矩阵 (课本模式)"],
        'vrp_params': "👇在此配置参数",
        'num_cust': "客户数量",
        'veh_cap': "车辆载重",
        'open_vrp': "车辆不回仓库 (Open VRP)",
        'open_vrp_hint': "勾选后，车辆送完最后一个客户直接下班。",
        'btn_plan': "🚀 立即规划路径",
        'no_solution': "无解 (可能载重不足)",
        # EOQ (修复部分)
        'eoq_title': "📦 库存控制中心",
        'tab1': "🧮 计算器",
        'tab2': "📖 公式原理",
        'D': "年需求量 (D)",
        'S': "单次订货成本 (S)",
        'H': "单位持有成本 (H)",
        'btn_calc': "计算 EOQ",
        'eoq_res': "最佳订货量",
        'eoq_desc': "该公式用于平衡订货成本与持有成本。",
        # Location
        'loc_title': "🏭 工厂选址与运输优化 (MIP)",
        'n_factories': "备选工厂数量",
        'n_customers': "客户数量",
        'cap_label': "最大产能",
        'fixed_cost': "建设成本",
        'dem_label': "需求量",
        'btn_loc_calc': "🚀 计算最优选址",
        'total_cost': "总综合成本",
        'trans_cost': "运输费用",
        'build_cost': "建设费用"
    },
    'en': {
        'title': "🚛 Logistics Master v5.1",
        'subtitle': "Integrated Platform for OR, Inventory & Routing",
        'sidebar_title': "⚙️ Control Panel",
        'modules': ["1. Vehicle Routing (VRP)", "2. EOQ Model", "3. Facility Location (MIP)"],
        # VRP
        'vrp_title': "🗺️ Vehicle Routing System",
        'vrp_mode': "Input Mode",
        'vrp_modes': ["Mode A: Coordinates", "Mode B: Distance Matrix"],
        'vrp_params': "👇 Parameters",
        'num_cust': "Customers",
        'veh_cap': "Vehicle Capacity",
        'open_vrp': "Open VRP",
        'open_vrp_hint': "No return to depot.",
        'btn_plan': "🚀 Optimize Routes",
        'no_solution': "Infeasible",
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
        'loc_title': "🏭 Facility Location (MIP)",
        'n_factories': "Potential Factories",
        'n_customers': "Customers",
        'cap_label': "Capacity",
        'fixed_cost': "Fixed Cost",
        'dem_label': "Demand",
        'btn_loc_calc': "🚀 Optimize Location",
        'total_cost': "Total Cost",
        'trans_cost': "Transport Cost",
        'build_cost': "Construction Cost"
    }
}
t = tr[st.session_state.language]

# --- 3. 顶部 Banner ---
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
    # 修复：确保 radio 的 options 和 index 逻辑正确
    selected_module_text = st.radio("Nav", t['modules'], label_visibility="collapsed")
    module_index = t['modules'].index(selected_module_text)
    st.markdown("---")

# ==================================================
# 模块 1: VRP
# ==================================================
def app_vrp():
    st.subheader(t['vrp_title'])
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
        if input_mode == t['vrp_modes'][0]: 
            if 'coord_df' not in st.session_state or len(st.session_state.coord_df) != num_customers + 1:
                init_data = {'x': [50]* (num_customers+1), 'y': [50]* (num_customers+1), 'demand': [10]* (num_customers+1)}
                init_data['demand'][0] = 0; init_data['x'][0] = 0; init_data['y'][0] = 0
                st.session_state.coord_df = pd.DataFrame(init_data)
                st.session_state.coord_df.index = ['W'] + [f'C{i}' for i in range(1, num_customers+1)]
            edited_df = st.data_editor(st.session_state.coord_df, key="editor_coords", use_container_width=True, height=200)
            coords = edited_df[['x', 'y']].values
            demands = edited_df['demand'].values
            n_total = len(coords)
            dist_matrix = np.zeros((n_total, n_total))
            for i in range(n_total):
                for j in range(n_total):
                    dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        else:
            n_total = num_customers + 1
            node_names = ['W'] + [f'C{i}' for i in range(1, n_total)]
            c_a, c_b = st.columns([1, 2])
            with c_a:
                init_demands = pd.DataFrame({'D': [0] + [10]*num_customers}, index=node_names)
                edited_demands = st.data_editor(init_demands, key="editor_demands", use_container_width=True, height=200)
                demands = edited_demands['D'].values
            with c_b:
                if 'dist_df' not in st.session_state or len(st.session_state.dist_df) != n_total:
                    st.session_state.dist_df = pd.DataFrame(np.zeros((n_total, n_total)), index=node_names, columns=node_names)
                edited_matrix = st.data_editor(st.session_state.dist_df, key="editor_matrix", use_container_width=True, height=200)
                dist_matrix = edited_matrix.values

    if st.button(t['btn_plan'], type="primary"):
        solve_dist_matrix = dist_matrix.copy()
        if is_open_vrp:
            for i in range(1, len(solve_dist_matrix)): solve_dist_matrix[i][0] = 0

        prob = pulp.LpProblem("VRP", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (range(n_total), range(n_total)), cat='Binary')
        u = pulp.LpVariable.dicts("u", range(n_total), 0, vehicle_cap, pulp.LpInteger)

        prob += pulp.lpSum([solve_dist_matrix[i][j] * x[i][j] for i in range(n_total) for j in range(n_total)])

        for i in range(1, n_total):
            prob += pulp.lpSum([x[i][j] for j in range(n_total) if i != j]) == 1
            prob += pulp.lpSum([x[j][i] for j in range(n_total) if i != j]) == 1
        
        for i in range(1, n_total):
            for j in range(1, n_total):
                if i != j: prob += u[i] - u[j] + vehicle_cap * x[i][j] <= vehicle_cap - demands[j]

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        if pulp.LpStatus[prob.status] == 'Optimal':
            st.success(f"Success! Total Distance: {pulp.value(prob.objective):.2f}")
            fig, ax = plt.subplots(figsize=(6, 4))
            G = nx.DiGraph()
            if coords is not None: pos = {i: (coords[i][0], coords[i][1]) for i in range(n_total)}
            else: pos = nx.circular_layout(range(n_total))
            
            nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='red', node_size=300, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=range(1,n_total), node_color='blue', node_size=200, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: (f"W" if i==0 else f"C{i}") for i in range(n_total)}, ax=ax, font_color='white', font_size=8)

            for i in range(n_total):
                for j in range(n_total):
                    if i != j and x[i][j].varValue > 0.5:
                        if is_open_vrp and j == 0: pass 
                        else: nx.draw_networkx_edges(G, pos, edgelist=[(i,j)], edge_color='black', width=1.5, ax=ax)
            st.pyplot(fig)
        else:
            st.error(t['no_solution'])

# ==================================================
# 模块 2: EOQ (修复 KeyError)
# ==================================================
def app_eoq():
    st.subheader(t['eoq_title'])
    
    # 确保这里用的 key 在字典里真实存在 (tab1, tab2)
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
            
            # 计算步骤
            numerator = 2 * D_val * S_val
            fraction = numerator / H_val
            
            st.info("💡 **Step-by-Step:**")
            st.markdown(f"""
            1. $2 \\times {D_val} \\times {S_val} = {numerator}$
            2. ${numerator} \\div {H_val} = {fraction}$
            3. $\\sqrt{{{fraction}}} \\approx {eoq}$
            """)
    
    with tab2:
        st.latex(r"Q^* = \sqrt{\frac{2DS}{H}}")
        st.caption(t['eoq_desc'])

# ==================================================
# 模块 3: 选址 (MIP)
# ==================================================
def app_location():
    st.subheader(t['loc_title'])
    
    c1, c2 = st.columns(2)
    num_factories = c1.slider(t['n_factories'], 1, 5, 3)
    num_customers = c2.slider(t['n_customers'], 1, 5, 3)
    
    factory_names = [f"F{i+1}" for i in range(num_factories)]
    customer_names = [f"D{j+1}" for j in range(num_customers)]
    
    col_f, col_d = st.columns(2)
    supply_data = {}
    fixed_cost_data = {}
    
    with col_f:
        for f in factory_names:
            c_cap, c_cost = st.columns(2)
            supply_data[f] = c_cap.number_input(f"{f} {t['cap_label']}", value=100, key=f"cap_{f}")
            fixed_cost_data[f] = c_cost.number_input(f"{f} {t['fixed_cost']}", value=5000, step=1000, key=f"cost_{f}")
            
    with col_d:
        demand_data = {}
        for d in customer_names:
            demand_data[d] = st.number_input(f"{d} {t['dem_label']}", value=60, key=f"dem_{d}")
            
    default_costs = [[10 + (i + j) * 2 for j in range(num_customers)] for i in range(num_factories)]
    cost_df = pd.DataFrame(default_costs, index=factory_names, columns=customer_names)
    edited_costs = st.data_editor(cost_df, use_container_width=True)
    
    if st.button(t['btn_loc_calc'], type="primary"):
        prob = pulp.LpProblem("Location", pulp.LpMinimize)
        flow = pulp.LpVariable.dicts("Flow", (factory_names, customer_names), 0, None, pulp.LpInteger)
        is_open = pulp.LpVariable.dicts("IsOpen", factory_names, cat='Binary')
        
        transport_cost = pulp.lpSum([flow[f][d] * edited_costs.loc[f, d] for f in factory_names for d in customer_names])
        build_cost = pulp.lpSum([is_open[f] * fixed_cost_data[f] for f in factory_names])
        prob += transport_cost + build_cost
        
        for d in customer_names:
            prob += pulp.lpSum([flow[f][d] for f in factory_names]) >= demand_data[d]
        for f in factory_names:
            prob += pulp.lpSum([flow[f][d] for d in customer_names]) <= supply_data[f] * is_open[f]
            
        prob.solve()
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            total = pulp.value(prob.objective)
            trans = pulp.value(transport_cost)
            build = pulp.value(build_cost)
            
            st.success("Optimal!")
            m1, m2, m3 = st.columns(3)
            m1.metric(t['total_cost'], f"{total:,.0f}")
            m2.metric(t['trans_cost'], f"{trans:,.0f}")
            m3.metric(t['build_cost'], f"{build:,.0f}")
            
            cols = st.columns(num_factories)
            opened = []
            for i, f in enumerate(factory_names):
                if is_open[f].varValue > 0.5:
                    cols[i].success(f"{f}: ✅")
                    opened.append(f)
                else:
                    cols[i].error(f"{f}: ❌")
                    
            G = nx.DiGraph()
            pos = {}
            for i, f in enumerate(factory_names):
                G.add_node(f, layer=0, status=('open' if f in opened else 'closed'))
                pos[f] = (0, -i*1.5)
            for i, d in enumerate(customer_names):
                G.add_node(d, layer=1)
                pos[d] = (2, -i*1.5)
            
            edge_labels = {}
            for f in factory_names:
                for d in customer_names:
                    val = flow[f][d].varValue
                    if val and val > 0:
                        G.add_edge(f, d)
                        edge_labels[(f,d)] = int(val)
            
            fig, ax = plt.subplots()
            color_map = ['gold' if G.nodes[n].get('status')=='open' else ('lightgrey' if G.nodes[n].get('status')=='closed' else 'lightgreen') for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=2000, ax=ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.25)
            st.pyplot(fig)
        else:
            st.error("Infeasible")

# --- 路由 ---
if module_index == 0:
    app_vrp()
elif module_index == 1:
    app_eoq()
elif module_index == 2:
    app_location()