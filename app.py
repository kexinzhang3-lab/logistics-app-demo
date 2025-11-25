# 决策变量：从工厂i运到目的地j的数量
flow = pulp.LpVariable.dicts("Route", (cost_matrix.index, cost_matrix.columns), 0, None, pulp.LpInteger)

# 目标函数：最小化总运费
prob += pulp.lpSum([flow[f][d] * edited_costs.loc[f, d] for f in cost_matrix.index for d in cost_matrix.columns])

# 约束条件：
# 1. 工厂运出量 <= 产能
for f in cost_matrix.index:
    prob += pulp.lpSum([flow[f][d] for d in cost_matrix.columns]) <= supply_data[f]

# 2. 目的地收到量 >= 需求
for d in cost_matrix.columns:
    prob += pulp.lpSum([flow[f][d] for f in cost_matrix.index]) >= demand_data[d]

# 求解
prob.solve()

# 5. 结果展示与绘图
if pulp.LpStatus[prob.status] == 'Optimal':
    st.success(f"找到最优方案！总费用: {pulp.value(prob.objective)}")
    
    # 准备绘图数据
    G = nx.DiGraph()
    pos = {}
    labels = {}
    edge_labels = {}
    
    # 设置节点位置：工厂在左(x=0)，目的地在右(x=1)
    for i, f in enumerate(cost_matrix.index):
        G.add_node(f, layer=0)
        pos[f] = (0, -i) # 纵向排列
    
    for j, d in enumerate(cost_matrix.columns):
        G.add_node(d, layer=1)
        pos[d] = (1, -j) # 纵向排列
        
    # 添加连线（只画有运输量的线）
    for f in cost_matrix.index:
        for d in cost_matrix.columns:
            amount = flow[f][d].varValue
            if amount > 0:
                G.add_edge(f, d)
                edge_labels[(f, d)] = f"{int(amount)}"

    # 画图
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, arrowsize=20, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    
    plt.axis('off')
    st.pyplot(fig)
    
else:
    st.error("无法找到可行解（可能总需求超过了总产能）")
